"""Summarization service — Phase 1 placeholder.

Phase 2 will implement:
  - ExtractiveSummarizer (TF-IDF + cosine similarity)
  - AbstractiveSummarizer (transformer / LLM API)
  - length control, chunking, structured summaries.

For Phase 1 this module only provides a clearly-labelled preview
(first few sentences) so the upload -> extraction -> NLP pipeline
can be demonstrated end to end.
"""


def placeholder_summary(sentences, max_sentences=3):
    """Return the first N sentences as a preview (NOT a real summary)."""
    sentences = sentences or []
    preview = [s.strip() for s in sentences[:max_sentences] if s and s.strip()]
    text = " ".join(preview).strip()
    if text:
        return text + "\n\n[Phase 1 preview — full extractive/abstractive summarization arrives in Phase 2.]"
    return "[Phase 1 preview — no sentences available for preview.]"
