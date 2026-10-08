"""Tests for the local extractive NLP summarization engine.

Extractive: short/medium/long docs, all lengths, ranking, redundancy
removal, original order, statistics.
Service: structured summary shape (extractive NLP only, fully offline).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nlp.extractive_summary import (  # noqa: E402
    ExtractiveSummarizer,
    extractive_summarize,
)
from services.summarization_service import (  # noqa: E402
    build_download_payload,
    summarize_document,
)


TOPICS = [
    "Natural language processing enables computers to understand human language.",
    "Machine learning models learn useful patterns from large text collections.",
    "TF-IDF weighting highlights terms that distinguish one sentence from others.",
    "Cosine similarity measures the angle between two sentence vectors.",
    "Neural networks capture complex nonlinear relationships in language data.",
    "Text classification assigns predefined labels to entire documents.",
    "Named entity recognition locates people, places and organizations in text.",
    "Sentiment analysis determines whether opinions expressed are positive or negative.",
    "Machine translation converts sentences from one language into another.",
    "Topic modelling discovers hidden themes across a document collection.",
    "Word embeddings represent vocabulary items as dense numeric vectors.",
    "Evaluation metrics such as precision and recall quantify model quality.",
]

SHORT_DOC = " ".join(TOPICS[:6])
MEDIUM_DOC = " ".join(TOPICS * 3)
LONG_DOC = " ".join(TOPICS * 12)


def _make_doc(text, sections=None):
    from nlp.keyword_extraction import extract_keywords
    from nlp.preprocessing import preprocess_text

    nlp = preprocess_text(text)
    return {
        "filename": "test.txt",
        "stored_path": "/tmp/test.txt",
        "file_type": "txt",
        "page_count": 1,
        "text": text,
        "metadata": {},
        "word_count": nlp["word_count"],
        "sentence_count": nlp["sentence_count"],
        "sentences": nlp["sentences"],
        "tokens": nlp["tokens"],
        "cleaned_text": nlp["cleaned_text"],
        "keywords": extract_keywords(text, sentences=nlp["sentences"], top_n=10),
        "sections": sections or [],
        "section_count": len(sections or []),
        "file_size_bytes": len(text),
    }


class TestExtractive(unittest.TestCase):
    def test_short_document(self):
        out = extractive_summarize(SHORT_DOC, length="short")
        self.assertTrue(out["summary"])
        self.assertLessEqual(len(out["selected_sentences"]), 5)
        self.assertGreater(out["summary_word_count"], 0)
        self.assertLess(out["summary_word_count"], out["original_word_count"])

    def test_medium_document_all_lengths(self):
        short = extractive_summarize(MEDIUM_DOC, length="medium")
        med = extractive_summarize(MEDIUM_DOC, length="medium")
        det = extractive_summarize(MEDIUM_DOC, length="detailed")
        sht = extractive_summarize(MEDIUM_DOC, length="short")
        self.assertEqual(short["summary"], med["summary"])  # deterministic
        self.assertLessEqual(
            len(sht["selected_sentences"]), len(det["selected_sentences"])
        )

    def test_detailed_selects_more_than_short(self):
        sht = extractive_summarize(LONG_DOC, length="short")
        det = extractive_summarize(LONG_DOC, length="detailed")
        self.assertLess(len(sht["selected_sentences"]), len(det["selected_sentences"]))

    def test_not_first_n_sentences(self):
        # The top-ranked sentence should rarely be exactly the first-N slice
        # on a shuffled document; at minimum the scorer must differentiate.
        out = extractive_summarize(MEDIUM_DOC, length="short")
        scores = [e["score"] for e in out["sentence_scores"]]
        self.assertGreater(len(set(scores)), 1)

    def test_original_order_preserved(self):
        unique_doc = " ".join(TOPICS)  # 12 distinct sentences
        out = extractive_summarize(unique_doc, length="medium")
        pos_by_sentence = {
            e["sentence"]: e["position"] for e in out["sentence_scores"]
        }
        order_in_summary = [
            pos_by_sentence[s] for s in out["selected_sentences"]
        ]
        # The summary lists selected sentences in document order.
        self.assertEqual(order_in_summary, sorted(order_in_summary))
        # And the joined summary respects that same order.
        spans = [out["summary"].index(s) for s in out["selected_sentences"]]
        self.assertEqual(spans, sorted(spans))

    def test_redundancy_removal(self):
        dup = (
            "Machine learning models learn patterns from data. " * 6
            + "TF-IDF weighting highlights distinctive sentence terms clearly. "
            + "Cosine similarity compares numeric sentence vectors for ranking. "
            + "Neural networks capture nonlinear relationships in language data. "
        )
        out = extractive_summarize(dup, length="detailed")
        lowered = [s.lower() for s in out["selected_sentences"]]
        self.assertEqual(len(lowered), len(set(lowered)))
        self.assertLessEqual(
            lowered.count("machine learning models learn patterns from data."), 1
        )

    def test_sentence_scores_structure(self):
        out = extractive_summarize(SHORT_DOC, length="medium")
        self.assertTrue(out["sentence_scores"])
        entry = out["sentence_scores"][0]
        self.assertIn("sentence", entry)
        self.assertIn("score", entry)
        self.assertIn("position", entry)
        scores = [e["score"] for e in out["sentence_scores"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_statistics_math(self):
        out = extractive_summarize(MEDIUM_DOC, length="medium")
        expected = 1.0 - out["summary_word_count"] / out["original_word_count"]
        self.assertAlmostEqual(out["compression_ratio"], round(expected, 4))

    def test_too_little_text(self):
        with self.assertRaises(ValueError):
            extractive_summarize("Too short.", length="short")
        with self.assertRaises(ValueError):
            ExtractiveSummarizer().summarize("", length="medium")

    def test_summarizer_class_reuse(self):
        s = ExtractiveSummarizer()
        a = s.summarize(SHORT_DOC, length="short")
        b = s.summarize(SHORT_DOC, length="short")
        self.assertEqual(a["summary"], b["summary"])


class TestService(unittest.TestCase):
    def test_extractive_structured_summary(self):
        sections = [
            {"title": "Introduction", "content": " ".join(TOPICS[:4])},
            {"title": "Methods", "content": " ".join(TOPICS[4:8])},
            {"title": "Results", "content": " ".join(TOPICS[8:12])},
        ]
        doc = _make_doc(MEDIUM_DOC, sections=sections)
        for length in ("short", "medium", "detailed"):
            result = summarize_document(doc, method="extractive", length=length)
            self.assertTrue(result["executive_summary"])
            self.assertTrue(result["key_points"])
            self.assertTrue(result["keywords"])
            self.assertTrue(result["section_summaries"])
            self.assertTrue(result["important_sentences"])
            stats = result["statistics"]
            for key in (
                "original_words", "summary_words", "compression_ratio",
                "original_sentences", "summary_sentences", "section_count",
                "keyword_count", "method", "length",
            ):
                self.assertIn(key, stats)
            self.assertEqual(stats["method"], "Extractive NLP")
            self.assertLessEqual(
                stats["summary_words"], stats["original_words"]
            )

    def test_no_sections_not_invented(self):
        doc = _make_doc(" ".join(TOPICS[:5]) + " Final closing statement wraps everything up neatly today.")
        result = summarize_document(doc, method="extractive", length="short")
        self.assertEqual(result["section_summaries"], [])

    def test_main_topics_capped(self):
        doc = _make_doc(MEDIUM_DOC)
        result = summarize_document(doc, method="extractive", length="medium")
        self.assertLessEqual(len(result["main_topics"]), 10)
        self.assertLessEqual(len(result["concepts"]), 8)

    def test_method_always_extractive(self):
        # Backward compat: any requested method maps to extractive NLP.
        doc = _make_doc(SHORT_DOC)
        result = summarize_document(doc, method="abstractive", length="short")
        self.assertEqual(result["method"], "extractive")
        self.assertEqual(result["statistics"]["method"], "Extractive NLP")

    def test_important_sentences_have_scores(self):
        doc = _make_doc(MEDIUM_DOC)
        result = summarize_document(doc, method="extractive", length="medium")
        for item in result["important_sentences"]:
            self.assertIn("sentence", item)
            self.assertIn("score", item)
            self.assertIn("position", item)

    def test_download_payload_structure(self):
        doc = _make_doc(MEDIUM_DOC)
        result = summarize_document(doc, method="extractive", length="medium")
        payload = build_download_payload(doc, result)
        for key in (
            "document_info", "summary_statistics", "executive_summary",
            "key_points", "section_summaries", "keywords",
            "important_concepts", "important_sentences", "conclusion",
        ):
            self.assertIn(key, payload)

    def test_too_little_text_service(self):
        doc = _make_doc("Tiny.")
        doc["sentences"] = ["Tiny."]
        doc["word_count"] = 1
        doc["sentence_count"] = 1
        with self.assertRaisesRegex(ValueError, "enough text"):
            summarize_document(doc, method="extractive", length="short")


if __name__ == "__main__":
    unittest.main()
