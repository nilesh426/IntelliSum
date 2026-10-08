"""Extractive summarization with TF-IDF + cosine similarity.

Pipeline (viva-friendly):
    sentences
        -> TF-IDF sentence vectors (scikit-learn)
        -> document centroid = mean of sentence vectors
        -> cosine similarity of each sentence to the centroid (relevance)
        -> blend with a small position weight (earlier sentences get a
           mild boost, since introductions often carry key context)
        -> rank sentences by score
        -> greedy redundancy removal (skip sentences too similar to one
           already chosen, measured with cosine similarity)
        -> restore original document order for readability

TF-IDF recap: a term scores high when it appears often in one sentence
(TF) but rarely across the document's sentences (IDF), so it spotlights
terms that distinguish a sentence. Cosine similarity measures the angle
between two vectors: 1 = same direction (very similar), 0 = unrelated.
"""

import re

from .preprocessing import segment_sentences

# Length presets: fraction of sentences to keep, with safety clamps so
# tiny documents still yield something readable and huge documents
# don't produce giant "summaries".
LENGTH_CONFIG = {
    "short": {"ratio": 0.12, "min_sentences": 2, "max_sentences": 5},
    "medium": {"ratio": 0.25, "min_sentences": 3, "max_sentences": 10},
    "detailed": {"ratio": 0.35, "min_sentences": 4, "max_sentences": 20},
}

# A document smaller than this cannot produce a meaningful summary.
MIN_SENTENCES = 3
MIN_WORDS = 30

# Redundancy cutoff: a candidate sentence whose cosine similarity to an
# already-selected sentence exceeds this is skipped as repetitive.
REDUNDANCY_THRESHOLD = 0.65


def _count_words(text):
    return len(re.findall(r"\b\w+\b", text or ""))


def _filter_sentences(sentences):
    """Keep substantive sentences; remember their original indices."""
    kept = []
    for idx, sent in enumerate(sentences or []):
        words = _count_words(sent)
        if 4 <= words <= 120:
            kept.append((idx, sent.strip()))
    return kept


def _sentence_vectors(texts):
    """Build TF-IDF vectors, one row per sentence."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",
        lowercase=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return matrix


def _score_sentences(matrix, n_sentences):
    """Score each sentence by relevance to the document centroid.

    relevance  = cosine(sentence_vector, centroid), where the centroid
                 (mean TF-IDF vector) represents the document's overall topic.
    position   = mild decay with sentence index (introductions matter).
    final      = 0.8 * relevance + 0.2 * position
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    centroid = np.asarray(matrix.mean(axis=0))
    relevance = cosine_similarity(matrix, centroid).ravel()
    position = np.array([1.0 / (1.0 + 0.05 * i) for i in range(n_sentences)])
    return (0.8 * relevance + 0.2 * position).tolist()


def _remove_redundant(ranked, matrix, keep):
    """Greedy selection: walk ranked candidates, skip near-duplicates.

    Two identical sentences have cosine similarity 1.0, so repeats are
    always filtered; paraphrases above REDUNDANCY_THRESHOLD are too.
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = dense / norms

    selected = []
    for cand_pos in ranked:
        if len(selected) >= keep:
            break
        if not selected:
            selected.append(cand_pos)
            continue
        sims = normalized[cand_pos] @ normalized[selected].T
        if float(np.max(sims)) < REDUNDANCY_THRESHOLD:
            selected.append(cand_pos)
    return selected


class ExtractiveSummarizer:
    """TF-IDF + cosine-similarity extractive summarizer."""

    def summarize(self, text, sentences=None, length="medium"):
        """Summarize text extractively.

        Returns:
            {
                "summary": str,
                "selected_sentences": list[str],
                "sentence_scores": [{"sentence": str, "score": float,
                                     "position": int}],
                "original_word_count": int,
                "summary_word_count": int,
                "compression_ratio": float,  # fraction removed, 0..1
            }

        Raises:
            ValueError: when the text is too short for a meaningful summary.
        """
        length = (length or "medium").lower()
        cfg = LENGTH_CONFIG.get(length, LENGTH_CONFIG["medium"])

        if sentences is None:
            sentences = segment_sentences(text or "")
        sentences = [s.strip() for s in (sentences or []) if s and s.strip()]

        original_words = _count_words(text)
        if len(sentences) < MIN_SENTENCES or original_words < MIN_WORDS:
            raise ValueError(
                "The document does not contain enough text to generate "
                "a meaningful summary."
            )

        kept = _filter_sentences(sentences)
        if len(kept) < MIN_SENTENCES:
            # Fall back to all sentences if filtering was too aggressive
            # (e.g. documents with very long or very short sentences).
            kept = [(i, s) for i, s in enumerate(sentences)]

        indices = [i for i, _ in kept]
        texts = [s for _, s in kept]
        n = len(texts)

        try:
            matrix = _sentence_vectors(texts)
        except ValueError as exc:
            raise ValueError(
                "The document does not contain enough text to generate "
                "a meaningful summary."
            ) from exc

        scores = _score_sentences(matrix, n)
        ranked = sorted(range(n), key=lambda p: scores[p], reverse=True)

        keep = min(n, max(cfg["min_sentences"], round(n * cfg["ratio"])))
        keep = min(keep, cfg["max_sentences"])
        # Always leave at least one sentence out when possible, so the
        # result is actually a summary rather than the full text.
        if n > cfg["min_sentences"]:
            keep = min(keep, n - 1)

        chosen = _remove_redundant(ranked, matrix, keep)
        if not chosen:
            chosen = ranked[: max(1, min(keep, n))]
        # Restore original document order so the summary reads naturally.
        chosen = sorted(chosen)

        selected = [texts[p] for p in chosen]
        summary = " ".join(selected).strip()
        summary_words = _count_words(summary)
        compression = (
            1.0 - (summary_words / original_words) if original_words else 0.0
        )

        sentence_scores = [
            {
                "sentence": texts[p],
                "score": round(float(scores[p]), 4),
                "position": indices[p],
            }
            for p in ranked
        ]

        return {
            "summary": summary,
            "selected_sentences": selected,
            "sentence_scores": sentence_scores,
            "original_word_count": original_words,
            "summary_word_count": summary_words,
            "compression_ratio": round(compression, 4),
        }


def extractive_summarize(text, sentences=None, length="medium"):
    """Convenience wrapper around :class:`ExtractiveSummarizer`."""
    return ExtractiveSummarizer().summarize(text, sentences=sentences, length=length)
