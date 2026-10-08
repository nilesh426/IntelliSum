"""PDF text extraction using PyMuPDF."""

import os

try:
    import pymupdf  # PyMuPDF >= 1.24 preferred namespace
except ImportError:  # pragma: no cover
    import fitz as pymupdf  # older PyMuPDF namespace


def extract_pdf(file_path):
    """Extract text, page count and metadata from a PDF file.

    Returns a consistent dict:
        {
            "filename": str,
            "file_type": "pdf",
            "page_count": int,
            "text": str,
            "metadata": dict,
            "pages": list[str],
        }

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file is corrupted/unreadable or empty.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    filename = os.path.basename(file_path)

    try:
        doc = pymupdf.open(file_path)
    except Exception as exc:
        raise ValueError(f"Could not read PDF file '{filename}'. It may be corrupted.") from exc

    try:
        page_count = doc.page_count
        pages = []
        for page in doc:
            try:
                pages.append(page.get_text("text") or "")
            except Exception:
                pages.append("")

        text = "\n\n".join(p.strip() for p in pages if p and p.strip())
        # Preserve original text (strip only leading/trailing blank space).
        text = text.strip()

        raw_meta = doc.metadata or {}
        metadata = {
            "title": raw_meta.get("title", "") or "",
            "author": raw_meta.get("author", "") or "",
            "subject": raw_meta.get("subject", "") or "",
            "producer": raw_meta.get("producer", "") or "",
            "format": raw_meta.get("format", "") or "",
        }
        # Drop empty metadata values.
        metadata = {k: v for k, v in metadata.items() if v}

        return {
            "filename": filename,
            "file_type": "pdf",
            "page_count": page_count,
            "text": text,
            "metadata": metadata,
            "pages": pages,
        }
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to extract text from PDF '{filename}'.") from exc
    finally:
        doc.close()
