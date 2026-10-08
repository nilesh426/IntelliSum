"""NLP preprocessing: normalization, segmentation, tokenization.

Keeps both ``original_text`` and ``processed_text`` so the source
document is never destroyed.
"""

import re
import unicodedata

# Small built-in stopword list used when NLTK data is unavailable.
_FALLBACK_STOPWORDS = frozenset(
    """
    a an and are as at be but by for from has have he her his i in is it its
    of on or that the their this to was were will with you your we our they
    them she him not no so if out up about into over after than then there
    these those such only also very can just do does did been had which who
    whom what when where why how all any both each few more most other some
    """.split()
)


def _get_nltk():
    try:
        import nltk  # noqa: WPS433

        return nltk
    except ImportError:
        return None


def normalize_unicode(text):
    """Apply NFKC unicode normalization."""
    return unicodedata.normalize("NFKC", text or "")


def normalize_whitespace(text):
    """Collapse all whitespace runs to single spaces and strip."""
    return re.sub(r"\s+", " ", text or "").strip()


def segment_sentences(text):
    """Split text into sentences.

    Tries NLTK ``sent_tokenize`` first, falls back to a regex splitter
    when NLTK or its punkt data is unavailable (e.g. offline machines).
    """
    text = (text or "").strip()
    if not text:
        return []

    nltk = _get_nltk()
    if nltk is not None:
        try:
            from nltk.tokenize import sent_tokenize

            return [s.strip() for s in sent_tokenize(text) if s.strip()]
        except LookupError:
            try:
                nltk.download("punkt", quiet=True)
                from nltk.tokenize import sent_tokenize

                return [s.strip() for s in sent_tokenize(text) if s.strip()]
            except Exception:
                pass
        except Exception:
            pass

    # Regex fallback: split after sentence-ending punctuation.
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def get_stopwords():
    """Return an English stopword set (NLTK preferred, fallback built-in)."""
    nltk = _get_nltk()
    if nltk is not None:
        try:
            from nltk.corpus import stopwords

            return set(stopwords.words("english"))
        except LookupError:
            try:
                nltk.download("stopwords", quiet=True)
                from nltk.corpus import stopwords

                return set(stopwords.words("english"))
            except Exception:
                pass
        except Exception:
            pass
    return set(_FALLBACK_STOPWORDS)


def tokenize_words(text, lower=True, remove_stopwords=True, remove_punct=True):
    """Tokenize text into cleaned word tokens.

    Steps: optional lowercasing, alphabetic filtering, stopword removal.
    """
    text = text or ""
    if lower:
        text = text.lower()

    nltk = _get_nltk()
    tokens = None
    if nltk is not None:
        try:
            from nltk.tokenize import word_tokenize

            tokens = word_tokenize(text)
        except LookupError:
            try:
                nltk.download("punkt", quiet=True)
                from nltk.tokenize import word_tokenize

                tokens = word_tokenize(text)
            except Exception:
                tokens = None
        except Exception:
            tokens = None

    if tokens is None:
        tokens = re.findall(r"[A-Za-z]+(?:'[a-z]+)?", text)

    stopwords = get_stopwords() if remove_stopwords else set()
    cleaned = []
    for tok in tokens:
        t = tok.lower() if lower else tok
        if remove_punct and not re.search(r"[A-Za-z0-9]", t):
            continue
        if remove_punct and not re.match(r"^[a-z0-9'\-]+$", t):
            # Drop stray punctuation tokens like '(', '``', '--'.
            if not re.search(r"[a-z0-9]", t):
                continue
        if len(t) < 2:
            continue
        if remove_stopwords and t in stopwords:
            continue
        cleaned.append(t)
    return cleaned


def lemmatize_tokens(tokens):
    """Lemmatize tokens with WordNet when available, else return as-is."""
    if not tokens:
        return []
    nltk = _get_nltk()
    if nltk is None:
        return list(tokens)
    try:
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        try:
            return [lemmatizer.lemmatize(t) for t in tokens]
        except LookupError:
            try:
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
                return [lemmatizer.lemmatize(t) for t in tokens]
            except Exception:
                return list(tokens)
    except Exception:
        return list(tokens)


def preprocess_text(text):
    """Run the full preprocessing pipeline.

    Returns:
        {
            "original_text": str,
            "cleaned_text": str,
            "sentences": list[str],
            "tokens": list[str],
            "word_count": int,
            "sentence_count": int,
        }
    """
    original = text or ""
    normalized = normalize_whitespace(normalize_unicode(original))
    sentences = segment_sentences(normalized)
    tokens = tokenize_words(normalized)
    tokens = lemmatize_tokens(tokens)
    cleaned_text = " ".join(tokens)
    word_count = len(re.findall(r"\b\w+\b", normalized)) if normalized else 0

    return {
        "original_text": original,
        "cleaned_text": cleaned_text,
        "sentences": sentences,
        "tokens": tokens,
        "word_count": word_count,
        "sentence_count": len(sentences),
    }
