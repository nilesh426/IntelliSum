import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Flask configuration for IntelliSum."""

    SECRET_KEY = os.environ.get("SECRET_KEY", "intellisum-dev-secret-key")

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.environ.get(
        "UPLOAD_FOLDER", os.path.join(BASE_DIR, "uploads")
    )
    OUTPUT_FOLDER = os.environ.get(
        "OUTPUT_FOLDER", os.path.join(BASE_DIR, "outputs")
    )

    # 16 MB default upload limit
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))

    ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

    # Phase 2 (abstractive) configuration — optional, read from .env
    LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
    LLM_MODEL = os.environ.get("LLM_MODEL", "")
