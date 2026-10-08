"""TXT text extraction with encoding fallbacks."""

import os


def extract_txt(file_path):
    """Read a plain-text file with UTF-8 and sensible fallbacks.

    Returns a consistent dict:
        {
            "filename": str,
            "file_type": "txt",
            "page_count": 1,
            "text": str,
            "metadata": dict,
        }

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file cannot be read.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    filename = os.path.basename(file_path)

    text = None
    used_encoding = None
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with open(file_path, "r", encoding=encoding) as fh:
                text = fh.read()
            used_encoding = encoding
            break
        except (UnicodeDecodeError, UnicodeError) as exc:
            last_error = exc
            continue
        except OSError as exc:
            raise ValueError(f"Could not read text file '{filename}'.") from exc

    if text is None:
        raise ValueError(f"Could not decode text file '{filename}'.") from last_error

    # Final safety net: replace undecodable characters instead of crashing.
    if not isinstance(text, str):
        raise ValueError(f"Could not decode text file '{filename}'.")

    text = text.strip()

    try:
        size_bytes = os.path.getsize(file_path)
    except OSError:
        size_bytes = 0

    return {
        "filename": filename,
        "file_type": "txt",
        "page_count": 1,
        "text": text,
        "metadata": {"encoding": used_encoding or "utf-8", "size_bytes": size_bytes},
    }
