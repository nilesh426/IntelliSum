"""Document upload validation + extraction + NLP orchestration (Phase 1)."""

import os
import uuid

from werkzeug.utils import secure_filename

from config import Config
from extraction.docx_extractor import extract_docx
from extraction.pdf_extractor import extract_pdf
from extraction.txt_extractor import extract_txt
from nlp.keyword_extraction import extract_keywords
from nlp.preprocessing import preprocess_text
from nlp.section_detection import detect_sections


def allowed_file(filename):
    """Check whether the filename has an allowed extension."""
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in Config.ALLOWED_EXTENSIONS


def get_file_type(filename):
    if not filename or "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()


def save_upload(file_storage, upload_folder):
    """Securely save an uploaded file and return its absolute path."""
    original = secure_filename(file_storage.filename or "")
    if not original:
        raise ValueError("No file selected.")
    unique = f"{uuid.uuid4().hex[:8]}_{original}"
    os.makedirs(upload_folder, exist_ok=True)
    dest = os.path.join(upload_folder, unique)
    file_storage.save(dest)
    # Reject empty uploads.
    if os.path.getsize(dest) == 0:
        os.remove(dest)
        raise ValueError("The uploaded file is empty.")
    return dest


def extract_text(file_path, file_type):
    """Dispatch to the correct extractor based on file type."""
    if file_type == "pdf":
        return extract_pdf(file_path)
    if file_type == "docx":
        return extract_docx(file_path)
    if file_type == "txt":
        return extract_txt(file_path)
    raise ValueError(f"Unsupported file type: '{file_type}'. Supported: PDF, DOCX, TXT.")


def process_document(file_path, file_type=None):
    """Extract text and run Phase-1 NLP (preprocessing, keywords, sections).

    Returns a dict with document info + NLP results.
    Raises ValueError with a user-friendly message on failure.
    """
    filename = os.path.basename(file_path)
    file_type = (file_type or get_file_type(file_path)).lower()

    if file_type not in Config.ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '.{file_type}'. Please upload PDF, DOCX or TXT."
        )

    result = extract_text(file_path, file_type)
    text = result.get("text", "") or ""

    if not text.strip():
        hint = ""
        if file_type == "pdf":
            hint = " The PDF may be scanned (images only) with no extractable text."
        raise ValueError(f"No extractable text found in '{filename}'.{hint}")

    nlp = preprocess_text(text)
    keywords = extract_keywords(text, sentences=nlp["sentences"], top_n=10)
    headings_hint = result.get("headings")
    sections = detect_sections(text, headings_hint=headings_hint)

    try:
        file_size = os.path.getsize(file_path)
    except OSError:
        file_size = 0

    return {
        "filename": filename,
        "stored_path": file_path,
        "file_type": file_type,
        "page_count": result.get("page_count"),
        "text": text,
        "metadata": result.get("metadata", {}),
        "word_count": nlp["word_count"],
        "sentence_count": nlp["sentence_count"],
        "sentences": nlp["sentences"],
        "tokens": nlp["tokens"],
        "cleaned_text": nlp["cleaned_text"],
        "keywords": keywords,
        "sections": sections,
        "section_count": len(sections),
        "file_size_bytes": file_size,
    }
