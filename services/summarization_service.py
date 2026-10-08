"""Summarization service: Flask <-> extractive NLP summarizer.

100% local pipeline (no LLM, no API, no network):

    Document text
        -> Text Preprocessing (Phase 1)
        -> Sentence Segmentation
        -> TF-IDF sentence vectors
        -> Cosine similarity to document centroid (relevance)
        -> Sentence Scoring (+ mild position weight)
        -> Redundancy Removal (cosine threshold)
        -> Important Sentence Selection
        -> Original Sentence Ordering
        -> Structured Summary

Builds the structured summary (document info, executive summary,
key points, section-wise summaries, keywords, concepts, important
sentences with scores, conclusion, statistics) without fabricating
content: every section comes from the document itself, and thin
sections are omitted.
"""

import re
import time
from collections import Counter

from nlp.extractive_summary import ExtractiveSummarizer, extractive_summarize
from nlp.preprocessing import segment_sentences

__all__ = [
    "ExtractiveSummarizer",
    "summarize_document",
    "build_download_payload",
]

VALID_LENGTHS = ("short", "medium", "detailed")

MAX_SECTION_SUMMARIES = 8
MAX_KEY_POINTS = 5
MAX_TOPICS = 8
MAX_CONCEPTS = 8

_FILLER_STARTS = frozenset(
    "the this that these those it they them he she we they".split()
)


def _count_words(text):
    return len(re.findall(r"\b\w+\b", text or ""))


def _extract_concepts(text, keywords, top_n=MAX_CONCEPTS):
    """Find important concept terms actually present in the document.

    Uses capitalized multi-word phrases (e.g. 'Machine Learning') plus
    top single-word keywords as a fallback. Never invents terms.
    """
    phrases = re.findall(
        r"\b([A-Z][a-zA-Z0-9\-]+(?:\s+[A-Z][a-zA-Z0-9\-]+){1,2})\b", text or ""
    )
    counts = Counter()
    for phrase in phrases:
        words = phrase.split()
        if words[0].lower() in _FILLER_STARTS:
            continue
        counts[phrase] += 1
    concepts = [p for p, _ in counts.most_common(top_n * 2)]

    if len(concepts) < top_n:
        for kw in keywords or []:
            if len(concepts) >= top_n:
                break
            if " " not in kw and kw not in concepts and len(kw) >= 4:
                concepts.append(kw)
    # De-duplicate case-insensitively, keep order.
    seen, unique = set(), []
    for concept in concepts:
        key = concept.lower()
        if key not in seen:
            seen.add(key)
            unique.append(concept)
    return unique[:top_n]


def _extract_conclusion(sentences):
    """Use the document's own closing sentence as the conclusion.

    Returns "" when no substantive closing sentence exists, so the UI
    can omit the section instead of fabricating one.
    """
    for sent in reversed(sentences or []):
        words = _count_words(sent)
        if 8 <= words <= 60:
            return sent.strip()
    return ""


def _build_extractive(doc, length):
    """Run the extractive pipeline incl. per-section summaries."""
    text = doc["text"]
    sentences = doc["sentences"]

    overall = extractive_summarize(text, sentences=sentences, length=length)

    # Section-wise: pick the best 1-3 sentences inside each section using
    # the same global sentence scores (factual, no generation).
    score_by_sentence = {
        entry["sentence"]: entry["score"]
        for entry in overall["sentence_scores"]
    }
    per_section = {"short": 1, "medium": 2, "detailed": 3}[length]
    section_summaries = []
    for section in (doc.get("sections") or [])[:MAX_SECTION_SUMMARIES]:
        sec_sents = [s for s in segment_sentences(section["content"]) if _count_words(s) >= 4]
        if not sec_sents:
            continue
        ranked = sorted(
            sec_sents, key=lambda s: score_by_sentence.get(s, 0.0), reverse=True
        )
        picks, seen = [], set()
        for sent in ranked:
            key = sent.lower()
            if key in seen:
                continue
            seen.add(key)
            picks.append(sent)
            if len(picks) >= per_section:
                break
        if picks:
            # Keep section-internal order for readability.
            order = {s: i for i, s in enumerate(sec_sents)}
            picks = sorted(picks, key=lambda s: order.get(s, 0))
            section_summaries.append(
                {"title": section["title"], "summary": " ".join(picks)}
            )

    # Key points: top-ranked sentences overall (rank order, not doc order).
    key_points = [
        entry["sentence"]
        for entry in overall["sentence_scores"][:MAX_KEY_POINTS]
    ]

    # Important sentences: selected sentences in document order, each
    # with its relevance score and original position. This powers the
    # "Selected Important Sentences" section and keeps the data
    # explainable (scores come straight from the TF-IDF ranker).
    score_lookup = {
        entry["sentence"]: entry for entry in overall["sentence_scores"]
    }
    important_sentences = []
    for sent in overall["selected_sentences"]:
        entry = score_lookup.get(sent, {})
        important_sentences.append(
            {
                "sentence": sent,
                "score": entry.get("score", 0.0),
                "position": entry.get("position", 0),
            }
        )

    return {
        "executive_summary": overall["summary"],
        "section_summaries": section_summaries,
        "key_points": key_points,
        "important_sentences": important_sentences,
        "sentence_scores": overall["sentence_scores"],
        "summary_word_count": overall["summary_word_count"],
        "summary_sentence_count": len(overall["selected_sentences"]),
        "compression_ratio": overall["compression_ratio"],
    }


def summarize_document(doc, method="extractive", length="medium"):
    """Build the full structured summary for a processed document.

    Args:
        doc: dict returned by ``document_service.process_document``.
        method: accepted for backward compatibility; always treated
            as "extractive" (no external AI is used).
        length: "short", "medium" or "detailed".

    Returns a structured summary dict with document info, statistics,
    executive summary, key points, section summaries, keywords,
    concepts, important sentences and conclusion. Raises ValueError
    with a user-friendly message when the text is too short.
    """
    # Local-only engine: any requested method maps to extractive.
    method = "extractive"
    length = (length or "medium").lower()
    if length not in VALID_LENGTHS:
        length = "medium"

    started = time.perf_counter()

    core = _build_extractive(doc, length)

    keywords = doc.get("keywords", []) or []
    main_topics = keywords[:MAX_TOPICS]
    concepts = _extract_concepts(doc.get("text", ""), keywords)
    conclusion = _extract_conclusion(doc.get("sentences", []))
    elapsed = round(time.perf_counter() - started, 2)

    compression = core["compression_ratio"]
    statistics = {
        "original_words": doc.get("word_count", 0),
        "summary_words": core["summary_word_count"],
        "compression_ratio": compression,
        "compression_percent": round(compression * 100, 2),
        "original_sentences": doc.get("sentence_count", 0),
        "summary_sentences": core["summary_sentence_count"],
        "section_count": doc.get("section_count", 0),
        "keyword_count": len(keywords),
        "method": "Extractive NLP",
        "length": length.capitalize(),
        "processing_time_sec": elapsed,
    }

    document_info = {
        "filename": doc.get("filename", ""),
        "file_type": (doc.get("file_type", "") or "").lower(),
        "file_size_bytes": doc.get("file_size_bytes", 0),
        "page_count": doc.get("page_count"),
        "word_count": doc.get("word_count", 0),
        "sentence_count": doc.get("sentence_count", 0),
        "section_count": doc.get("section_count", 0),
        "keyword_count": len(keywords),
    }

    return {
        "method": method,
        "length": length,
        "document_info": document_info,
        "executive_summary": core["executive_summary"],
        "main_topics": main_topics,
        "section_summaries": core["section_summaries"],
        "key_points": core["key_points"],
        "keywords": keywords,
        "concepts": concepts,
        "important_concepts": concepts,
        "important_sentences": core["important_sentences"],
        "sentence_scores": core["sentence_scores"],
        "conclusion": conclusion,
        "statistics": statistics,
        "summary_statistics": statistics,
    }


def build_download_payload(doc, result):
    """Assemble a clean structured payload for Phase 3 PDF/DOCX export.

    Uses the same structured data rendered on the result page, so the
    export step can consume it without recomputation.
    """
    stats = result.get("statistics", {}) or {}
    doc_info = result.get("document_info", {}) or {
        "filename": doc.get("filename", ""),
        "file_type": (doc.get("file_type", "") or "").lower(),
        "file_size_bytes": doc.get("file_size_bytes", 0),
        "page_count": doc.get("page_count"),
        "word_count": doc.get("word_count", 0),
        "sentence_count": doc.get("sentence_count", 0),
        "section_count": doc.get("section_count", 0),
        "keyword_count": len(result.get("keywords", []) or []),
    }
    return {
        "document_info": doc_info,
        "summary_statistics": stats,
        "executive_summary": result.get("executive_summary", ""),
        "key_points": result.get("key_points", []),
        "section_summaries": result.get("section_summaries", []),
        "keywords": result.get("keywords", []),
        "important_concepts": result.get("important_concepts", result.get("concepts", [])),
        "important_sentences": result.get("important_sentences", []),
        "conclusion": result.get("conclusion", ""),
    }
