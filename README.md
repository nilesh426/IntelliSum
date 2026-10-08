# IntelliSum — Intelligent Document Summarization and Structuring System

> **Local NLP edition** — 100% offline extractive summarizer (TF-IDF +
> cosine similarity), length control, structured professional result page,
> and length-only resummarize UI. No LLM, no API key, no network needed
> for summarization.
> (PDF/DOCX export arrives in Phase 3.)

## Problem Statement

Long documents (reports, papers, manuals) are time-consuming to read.
IntelliSum accepts PDF, DOCX and TXT files and produces a meaningful,
well-structured summary using traditional NLP techniques — fully local,
no external AI API.

## Features (Phase 1 + Local NLP Engine)

### Document intake (Phase 1)

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

### Summarization engine (Extractive NLP, fully local)

- **Extractive NLP** (`nlp/extractive_summary.py`): sentence segmentation →
  reused Phase 1 preprocessing → TF-IDF sentence vectors (scikit-learn) →
  cosine similarity to document centroid + mild position weight → ranking →
  greedy redundancy removal (cosine threshold) → original order restored.
  Never just takes the first N sentences.
- **Lengths**: Short (quick overview, highest-ranked sentences only),
  Medium (balanced, major information), Detailed (comprehensive, additional
  important sentences), with min/max clamps so tiny and huge docs stay sensible.
- **Structured summary**: Document Information, Summary Overview, Executive
  Summary (top-ranked original sentences), Key Points, Section-wise Summary,
  Important Keywords (badges), Important Concepts, Selected Important
  Sentences (numbered + relevance scores), Summary Statistics
  (words, compression %, sentences, sections, keywords, method, length,
  time), Conclusion (document's own closing; omitted if none), and
  "How the Summary Was Generated" (viva-friendly NLP explanation).
- **UI**: professional result page with stat cards, section cards, keyword
  badges and numbered sentence lists; "Try a different summary length" form
  (`/resummarize`) without re-uploading; duplicate submits blocked.
- **Download-ready data**: `build_download_payload()` returns
  `document_info / summary_statistics / executive_summary / key_points /
  section_summaries / keywords / important_concepts / important_sentences /
  conclusion` for Phase 3 PDF/DOCX export.

Strictly excluded (per project scope): external LLMs, API keys, RAG,
vector DBs, FAISS/ChromaDB, embeddings, semantic retrieval,
document Q&A, chatbots.

## Tech Stack

Python, Flask, Jinja2, HTML/CSS/JS, PyMuPDF, python-docx, NLTK,
scikit-learn (TF-IDF, cosine similarity), python-dotenv.

## Project Structure

```
IntelliSum/
├── app.py                  # Flask app + routes (/summarize, /resummarize)
├── config.py               # configuration (uploads, limits, .env)
├── requirements.txt
├── .env.example
├── uploads/  outputs/
├── extraction/             # pdf_extractor, docx_extractor, txt_extractor
├── nlp/                    # preprocessing, extractive_summary,
│                           # keyword_extraction, section_detection
├── services/               # document_service, summarization_service, output_service
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
copy .env.example .env        # Windows (optional — defaults work without it)
# cp .env.example .env       # Linux/macOS
```

## Running the Project

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in a browser.
No API key or internet connection is required for summarization.

## Configuration (.env, optional)

| Variable | Default | Purpose |
|---|---|---|
| `SECRET_KEY` | dev key | Flask sessions/flash |
| `MAX_CONTENT_LENGTH` | 16777216 (16 MB) | upload size limit |

## Testing

```bash
python -m unittest discover -s tests -v
```

Covers: PDF/DOCX/TXT upload, invalid/empty/corrupt files, extraction,
preprocessing, keywords, section detection, extractive (lengths, ranking,
redundancy, order, stats, important sentences, download payload),
routes plus `/resummarize`. Fully offline.

## Limitations

- Scanned/image-only PDFs yield no text (reported as a friendly error).
- Documents under ~3 sentences / 30 words are rejected as too short.
- Section detection is heuristic; unstructured prose returns no sections
  (the section-wise block is then hidden with a friendly note).

## Future Scope (Phase 3)

ReportLab PDF + python-docx downloads (via `build_download_payload`),
UI polish, full academic documentation, final testing.
