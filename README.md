# IntelliSum — Intelligent Document Summarization and Structuring System

> **Phase 1** — Flask foundation + document upload + PDF/DOCX/TXT extraction +
> NLP preprocessing + TF-IDF keyword extraction + section detection.
> (Extractive/abstractive summarization, downloads and full docs arrive in Phases 2–3.)

## Problem Statement

Long documents (reports, papers, manuals) are time-consuming to read.
IntelliSum accepts PDF, DOCX and TXT files and produces a meaningful,
well-structured summary using real NLP techniques — not just an AI API wrapper.

## Features (Phase 1)

- File upload: PDF, DOCX, TXT (extension validation, 16 MB limit,
  secure filenames, empty/corrupt/unsupported handling; uploads never executed)
- Text extraction
  - PDF via PyMuPDF (text, page count, page-wise text, metadata)
  - DOCX via python-docx (paragraphs, headings, table text)
  - TXT via UTF-8 with fallback decoding
- NLP preprocessing (unicode + whitespace normalization, sentence
  segmentation, tokenization, stopword/punctuation handling, lemmatization;
  keeps `original_text` and `processed_text`, returns sentences, tokens,
  cleaned text, word/sentence counts)
- Keyword extraction (TF-IDF over sentences + frequency fallback,
  n-grams, ~5–15 cleaned keyphrases)
- Section detection (numbered/markdown/ALL-CAPS/label headings, DOCX heading
  hints; returns [] rather than inventing sections)
- Web UI (upload form, length + method selectors, analysis result page)

Strictly excluded (per project scope): RAG, vector DBs, FAISS/ChromaDB,
document Q&A, chatbots, semantic search, retrieval pipelines.

## Tech Stack

Python, Flask, Jinja2, HTML/CSS/JS, PyMuPDF, python-docx, NLTK,
scikit-learn (TF-IDF, cosine similarity in Phase 2), python-dotenv.

## Project Structure

```
IntelliSum/
├── app.py                  # Flask app + routes
├── config.py               # configuration (uploads, limits, .env)
├── requirements.txt
├── .env.example
├── uploads/  outputs/
├── extraction/             # pdf_extractor, docx_extractor, txt_extractor
├── nlp/                    # preprocessing, keyword_extraction, section_detection
├── services/               # document_service (+ Phase 2/3 placeholders)
├── templates/              # base, index, result
├── static/css/style.css  static/js/app.js
└── tests/                  # test_extraction, test_nlp, test_summarization, test_routes
```

## Installation

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
copy .env.example .env        # Windows
# cp .env.example .env       # Linux/macOS
```

## Running the Project

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in a browser.

## Configuration (.env)

| Variable | Default | Purpose |
|---|---|---|
| `SECRET_KEY` | dev key | Flask sessions/flash |
| `MAX_CONTENT_LENGTH` | 16777216 (16 MB) | upload size limit |
| `LLM_API_KEY` / `LLM_MODEL` | empty | Phase 2 abstractive backend (optional) |

## Testing

```bash
python -m unittest discover -s tests -v
```

Covers: PDF/DOCX/TXT upload, invalid/empty/corrupt files, extraction,
preprocessing, keywords, section detection. (`test_summarization.py` is a
Phase-2 placeholder and skips.)

## Limitations (Phase 1)

- Scanned/image-only PDFs yield no text (reported as a friendly error).
- Section detection is heuristic; unstructured prose returns no sections.
- Result page shows a labelled preview — full summarization is Phase 2.

## Future Scope

Phase 2: extractive (TF-IDF + cosine) + abstractive summarizers, length
control, chunking, structured summaries.
Phase 3: ReportLab PDF + python-docx downloads, UI polish, full docs.
