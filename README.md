# IntelliSum
### Intelligent Document Summarization and Structuring System

> **Traditional / explainable NLP-based extractive document summarization system.**
> It does not use an external LLM API. It does not use RAG. It does not use a
> vector database. It does not use document retrieval, embeddings, or a chatbot.
> Summarization works fully offline once dependencies are installed.

---

## 1. Project Overview

IntelliSum is a web-based academic NLP mini-project that accepts **PDF, DOCX and
TXT** documents and produces a meaningful, well-structured **extractive summary**
using traditional, explainable NLP techniques: text extraction, preprocessing,
sentence segmentation, **TF-IDF**, **cosine similarity**, sentence scoring,
redundancy removal, keyword/keyphrase extraction, section detection and document
statistics. The structured summary is displayed on a professional result page and
can be exported as **PDF** (ReportLab) or **DOCX** (python-docx).

## 2. Problem Statement

Long documents such as reports, research papers and manuals are time-consuming to
read in full. Readers need a quick, trustworthy overview that preserves the
document's own wording and factual content — without sending private documents to
an external AI service.

## 3. Motivation

- Save reading time with concise, structured overviews of long documents.
- Keep documents private: everything runs locally, no API key, no uploads to
  third-party AI services.
- Stay explainable: every sentence in the summary comes from the source
  document, and sentence relevance scores show *why* it was selected — ideal for
  viva/defense questions.

## 4. Objectives

1. Extract clean text from PDF, DOCX and TXT files.
2. Preprocess text with standard NLP steps (normalization, segmentation,
   tokenization, stopword handling, lemmatization).
3. Rank sentences with TF-IDF + cosine similarity and select the most important
   ones extractively.
4. Extract keywords/keyphrases and detect document sections.
5. Present a structured summary (executive summary, key points, section-wise
   summary, keywords, concepts, important sentences, statistics, conclusion).
6. Export the summary as PDF and DOCX with filenames derived from the source
   document.
7. Handle errors gracefully without exposing internals.

## 5. Key Features

- **Document intake**: PDF / DOCX / TXT upload with extension validation, a
  16 MB size limit, secure filenames, and friendly handling of empty, corrupt,
  unsupported, too-short and scanned (image-only) files. Uploaded files are
  never executed.
- **Extractive summarization** with Short / Medium / Detailed lengths.
- **Structured result page**: document information, summary overview stat cards,
  executive summary, key points, section-wise summaries, keyword badges,
  important concepts, numbered important sentences with relevance scores,
  summary statistics, conclusion, and a "How the Summary Was Generated"
  explanation.
- **Length-only resummarization** (`/resummarize`) without re-uploading.
- **PDF / DOCX export** (`Download PDF` / `Download DOCX`) generated from the
  *same* structured data as the webpage, so numbers and text always match.
- **Responsive UI** for desktop, laptop, tablet and mobile (no frontend
  framework — plain HTML/CSS/JS).
- **Staged loading indicator** ("Extracting text… Analyzing sentences…
  Calculating sentence importance…") with duplicate-submission prevention.

## 6. Supported Documents

| Format | Extraction | Notes |
|---|---|---|
| PDF | PyMuPDF (text per page, page count, metadata) | Scanned/image-only PDFs report "no extractable text" (no OCR in this version) |
| DOCX | python-docx (paragraphs, heading styles, table text, core properties) | Headings feed section detection |
| TXT | UTF-8-SIG → UTF-8 → Latin-1 fallback decoding | Page count reported as 1 |

Limits: 16 MB upload cap (configurable via `MAX_CONTENT_LENGTH`); documents need
at least ~3 sentences / 30 words for a meaningful summary.

## 7. Technology Stack

| Layer | Technology |
|---|---|
| Web | Python, Flask, Jinja2, HTML/CSS/JS |
| PDF extraction | PyMuPDF |
| DOCX read/write | python-docx |
| NLP | NLTK (segmentation/tokenization/stopwords/lemmatization with offline fallbacks), scikit-learn (TF-IDF, cosine similarity) |
| PDF export | ReportLab (Platypus flowables) |
| Config | python-dotenv (optional `.env`) |
| Testing | unittest (stdlib) |

No LLM SDKs, no vector-database packages, no embedding libraries, no retrieval
pipelines.

## 8. System Architecture

```text
Browser (index.html / result.html / app.js + style.css)
        │  upload / resummarize / download
        ▼
app.py (Flask routes: /summarize, /resummarize, /download/<pdf|docx>)
        │
        ├── services/document_service.py  (validation → extraction → Phase-1 NLP)
        │       ├── extraction/pdf_extractor.py | docx_extractor.py | txt_extractor.py
        │       └── nlp/preprocessing.py + keyword_extraction.py + section_detection.py
        │
        ├── services/summarization_service.py  (extractive engine → structured result)
        │       └── nlp/extractive_summary.py  (TF-IDF + cosine scoring)
        │
        └── services/output_service.py  (build_pdf / build_docx from the same payload)
```

## 9. System Workflow

```text
Document
   ↓
Text Extraction
   ↓
Text Preprocessing
   ↓
Sentence Segmentation
   ↓
TF-IDF (sentence vectors)
   ↓
Sentence Scoring (cosine similarity to document centroid + position weight)
   ↓
Redundancy Removal (pairwise cosine threshold)
   ↓
Important Sentence Selection
   ↓
Original Sentence Ordering
   ↓
Structured Summary (webpage + PDF/DOCX export)
```

## 10. Document Processing

`services/document_service.py` orchestrates intake: `secure_filename()` +
extension check → save with a uuid prefix into `uploads/` → dispatch to the
correct extractor → reject empty/unreadable results with friendly messages →
run Phase-1 NLP (preprocessing, keywords, sections) → return a document dict
(`filename` shown to the user is the *original* name; the uuid prefix is
internal only).

## 11. NLP Preprocessing

`nlp/preprocessing.py`: NFKC unicode normalization → whitespace collapsing →
sentence segmentation (NLTK `sent_tokenize`, regex fallback when punkt data is
unavailable) → word tokenization (NLTK with regex fallback) → lowercasing,
punctuation/short-token filtering, stopword removal (NLTK list with a built-in
fallback list) → WordNet lemmatization when available. Both `original_text` and
`cleaned_text` are kept, plus sentences, tokens and word/sentence counts.

## 12. Keyword Extraction

`nlp/keyword_extraction.py` treats each sentence as a mini-document and scores
unigrams/bigrams with TF-IDF (`ngram_range=(1,2)`, up to 500 features), falling
back to frequency analysis (stopword-filtered unigrams + repeated bigrams) when
TF-IDF cannot run. Candidates are lowercased, de-duplicated (including
substring duplicates) and capped to 5–15 keyphrases (`top_n=10` by default).

## 13. Section Detection

`nlp/section_detection.py` is purely heuristic and **never invents headings**:
numbered headings (`1. Introduction`, `IV. …`), Markdown `#` headings, ALL-CAPS
lines, short `Label:` lines, DOCX heading-style hints, and short Title-Case
lines surrounded by blank space. Fewer than 2 reliable headings → returns `[]`,
and the UI/export show "No reliable document sections were detected."

## 14. Extractive Summarization

`nlp/extractive_summary.py`: the system **does not create new sentences — it
selects the most important sentences from the original document.** Pipeline:

1. Filter out trivial sentences (< 4 or > 120 words).
2. Build one TF-IDF vector per sentence.
3. Compute the document **centroid** (mean of sentence vectors).
4. Score each sentence: `final = 0.8 × relevance + 0.2 × position`, where
   relevance is cosine similarity to the centroid and position is a mild
   `1 / (1 + 0.05 × index)` boost for earlier sentences.
5. Rank, then greedily skip sentences whose cosine similarity to an
   already-chosen sentence exceeds **0.65** (redundancy removal).
6. Restore original document order and join into the summary.

## 15. TF-IDF

TF-IDF (Term Frequency × Inverse Document Frequency) assigns higher importance
to words that are frequent in one sentence but rare across the document:

```text
TF = (occurrences of term in sentence) / (terms in sentence)
IDF = log(total sentences / sentences containing the term)
TF-IDF = TF × IDF
```

Here each *sentence* plays the role of a "document": terms that distinguish one
sentence from the rest (e.g. `TF-IDF`, `classification`) get high weights, while
common words get near-zero weights (English stopwords are also removed). The
project uses scikit-learn's `TfidfVectorizer` over sentence vectors.

## 16. Cosine Similarity

Cosine similarity measures the angle between two vector representations:

```text
cosine similarity = (A · B) / (||A|| × ||B||)
```

`1` = same direction (very similar), `0` = unrelated. IntelliSum uses it twice:
(1) each sentence vector vs. the **document centroid** to measure relevance to
the overall topic, and (2) candidate vs. already-selected sentences to **reduce
redundancy** (near-duplicates score close to 1 and are skipped).

## 17. Sentence Scoring

Final score per sentence: **80% topical relevance** (cosine to centroid) plus a
**20% position prior** favouring earlier sentences, since introductions often
carry key context. Scores are stored per sentence (`sentence_scores`) and shown
beside each selected sentence on the result page, making the ranking
explainable in a viva.

## 18. Redundancy Removal

Candidates are visited in rank order; a candidate whose cosine similarity to any
already-selected sentence is ≥ 0.65 is skipped. Identical repeated sentences
score 1.0, so exact duplicates can never appear twice. Selection stops at the
length budget (always leaving at least one sentence out when possible, so the
output is a summary, not the full text).

## 19. Summary Lengths

| Length | Sentence fraction | Min–max sentences | Meaning |
|---|---|---|---|
| Short | ~12% | 2–5 | Quick overview, only the highest-ranked sentences |
| Medium | ~25% | 3–10 | Balanced summary with the major information |
| Detailed | ~35% | 4–20 | Comprehensive summary with additional important sentences |

Documents under ~3 sentences / 30 words are rejected as too short.

## 20. Large Document Handling

Extractive summarization needs no chunking: TF-IDF over all sentence vectors is
cheap (one sparse matrix + centroid), so even long documents are processed in a
single pass with no truncation, no API calls and no cost. Section-wise
summaries reuse the same global scores (1/2/3 sentences per section for
Short/Medium/Detailed).

## 21. Structured Summary Generation

`summarize_document()` returns one dict used by the webpage *and* both exports:

```python
{
    "document_info": {...},       # filename, type, size, pages, counts
    "summary_statistics": {...},  # words, compression %, sentences, ...
    "executive_summary": "...",   # top-ranked sentences, original order
    "key_points": [...],          # top-ranked sentences, rank order
    "section_summaries": [...],   # per-section extractive picks
    "keywords": [...],
    "important_concepts": [...],
    "important_sentences": [...], # selected sentences + scores + positions
    "conclusion": "...",          # document's own closing sentence (or "")
}
```

`build_download_payload()` exposes exactly this shape for the export layer.
Nothing is calculated separately in the route, the HTML, or the exporters.

## 22. Summary Statistics

All values are computed dynamically per document: original/summary words,
compression ratio (`1 − summary/original`, shown as %), original/selected
sentence counts, sections detected, keywords extracted, method
(`Extractive NLP`), length and processing time. Statistics on the page, in the
PDF table and in the DOCX table are identical because they share one payload.

## 23. PDF/DOCX Export

`services/output_service.py` implements `build_pdf()` (ReportLab Platypus:
`SimpleDocTemplate`, `Paragraph`, `Table`/`TableStyle`, `ListFlowable`,
`Spacer`, page numbers, A4 with 2 cm margins) and `build_docx()` (python-docx
`Title`/`Heading 1`/`Heading 2`/`Normal`/`List Bullet`/`List Number` styles plus
info tables). Content mirrors the webpage sections; empty sections are omitted.
Routes: `GET /download/pdf?doc_id=…&length=…` and `GET /download/docx?...`,
served as attachments named `<original>_summary_<length>.pdf/.docx`
(sanitized via `secure_filename`, contained in `outputs/`).

## 24. Project Structure

```text
IntelliSum/
├── app.py                  # Flask app + routes (/summarize, /resummarize, /download/<pdf|docx>)
├── config.py               # uploads, outputs, 16 MB limit, .env
├── requirements.txt
├── README.md
├── .gitignore / .env.example
├── uploads/  outputs/      # runtime files (git-kept empty; stale exports auto-cleaned)
├── extraction/             # pdf_extractor, docx_extractor, txt_extractor
├── nlp/                    # preprocessing, extractive_summary, keyword_extraction, section_detection
├── services/               # document_service, summarization_service, output_service
├── templates/              # base, index, result
├── static/css/style.css  static/js/app.js
└── tests/                  # test_extraction, test_nlp, test_summarization, test_routes, test_export
```

## 25. Installation

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

Dependencies actually used: `Flask`, `PyMuPDF`, `python-docx`, `nltk`,
`scikit-learn`, `python-dotenv`, `reportlab`.

## 26. Running the Application

```bash
python app.py
```

Then open **http://127.0.0.1:5000**. Upload a PDF/DOCX/TXT → pick a length →
Generate Summary → optionally resummarize at another length or Download PDF /
DOCX. No API key or internet connection is required for summarization.

## 27. Testing

```bash
python -m unittest discover -s tests -v
```

60 tests, fully offline: extraction (PDF/DOCX/TXT, corrupt, missing),
preprocessing/keywords/sections, extractive engine (lengths, ranking,
redundancy, order, stats, download payload), routes (upload validation,
resummarize, error cases), and exports (PDF validity/content/pagination, DOCX
structure/content, empty-section omission, filename safety, download routes for
all formats × lengths, webpage/export consistency, leak-free error pages).
Generated sample files are opened and read back in the tests (PDF via PyMuPDF,
DOCX via python-docx).

## 28. Advantages

- 100% local and private — documents never leave the machine.
- Explainable — every summary sentence is traceable to the source with a score.
- Deterministic — same document + length always yields the same summary.
- Zero inference cost, no quotas, works offline.
- Honest output — no hallucinated facts, no invented headings.

## 29. Limitations

- Extractive summaries reuse original sentences; the system cannot generate new
  paraphrased sentences.
- Statistical sentence ranking may miss deep semantic meaning that a large
  language model would capture.
- Scanned/image-only PDFs require OCR, which is not included.
- Section detection depends on document formatting; poorly formatted documents
  may yield weaker section grouping (content summary still works).
- Very short documents (< ~3 sentences / 30 words) cannot be summarized.

## 30. Future Scope

- OCR for scanned PDFs (e.g. Tesseract).
- Multilingual summarization.
- Improved semantic sentence ranking.
- Transformer-based (abstractive) summarization as an *optional* add-on.
- Better document structure detection and sentence-importance visualization.
- Additional document formats (e.g. HTML, EPUB).

These are future scope only and are **not** implemented in this version.

## 31. Conclusion

IntelliSum demonstrates a complete, traditional NLP pipeline — from raw PDF,
DOCX and TXT files to a structured, explainable extractive summary with
statistics and one-click PDF/DOCX export — using only local processing
(TF-IDF + cosine similarity) and a clean Flask web interface suitable for
academic demonstration and viva defense.
