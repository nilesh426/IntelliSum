"""Explainable keyword extraction using TF-IDF + frequency analysis."""

import re
from collections import Counter

from .preprocessing import segment_sentences


def _tfidf_keywords(text, sentences, top_n):
    """Score terms with TF-IDF treating sentences as documents."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = [s for s in (sentences or []) if s and len(s.split()) >= 3]
    if len(docs) < 2:
        # Fall back to paragraph chunks so TF-IDF has >1 document.
        paras = [p.strip() for p in re.split(r"\n{2,}|\n", text or "") if p.strip()]
        docs = [p for p in paras if len(p.split()) >= 3]
    if len(docs) < 2:
        return None

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=500,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",
        lowercase=True,
    )
    try:
        matrix = vectorizer.fit_transform(docs)
    except ValueError:
        return None  # e.g. empty vocabulary

    import numpy as np

    mean_scores = np.asarray(matrix.mean(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, mean_scores), key=lambda x: x[1], reverse=True)
    return [term for term, _score in ranked[: max(top_n * 3, top_n)]]


def _frequency_keywords(text, top_n):
    """Simple frequency-based keyword fallback."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", (text or "").lower())
    stop = set(ENGLISH_STOP_WORDS)
    filtered = [w.strip("-") for w in words if w not in stop and len(w) >= 3]
    # Unigrams.
    counts = Counter(filtered)
    # Bigrams of filtered words (consecutive meaningful words).
    bigrams = [" ".join(pair) for pair in zip(filtered, filtered[1:])]
    bigram_counts = Counter(b for b in bigrams if len(b.split()) == 2)
    # Merge: weight bigrams slightly so key phrases surface.
    scored = Counter()
    scored.update(counts)
    for phrase, count in bigram_counts.most_common(top_n * 2):
        if count >= 2:
            scored[phrase] = count * 2
    return [term for term, _ in scored.most_common(top_n * 3)]


def _clean_terms(candidates, top_n):
    """Lowercase, drop meaningless terms, de-duplicate."""
    seen = set()
    results = []
    for term in candidates or []:
        t = (term or "").strip().lower()
        t = re.sub(r"\s+", " ", t)
        if not t or len(t) < 3:
            continue
        if re.fullmatch(r"[\d\W_]+", t):
            continue
        if t in seen:
            continue
        # Drop terms that are substrings of an already accepted longer term.
        if any(t in existing for existing in results):
            continue
        # Drop single chars / pure digits.
        if len(t.split()) == 1 and (t.isdigit() or len(t) < 3):
            continue
        seen.add(t)
        results.append(t)
        if len(results) >= top_n:
            break
    return results


def extract_keywords(text, sentences=None, top_n=10):
    """Extract ~5-15 important keywords/keyphrases from text.

    Uses TF-IDF over sentences (explainable) with a frequency fallback.
    Never raises on short/odd input — returns [] instead.
    """
    top_n = max(5, min(15, int(top_n or 10)))
    text = text or ""
    if not text.strip():
        return []

    if sentences is None:
        sentences = segment_sentences(text)

    candidates = None
    try:
        candidates = _tfidf_keywords(text, sentences, top_n)
    except Exception:
        candidates = None

    if not candidates:
        try:
            candidates = _frequency_keywords(text, top_n)
        except Exception:
            return []

    return _clean_terms(candidates, top_n)
