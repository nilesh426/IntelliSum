"""IntelliSum — Intelligent Document Summarization and Structuring System.

100% local NLP summarization engine (no LLM, no API, no network):
text extraction -> preprocessing -> sentence segmentation -> TF-IDF
-> sentence scoring (cosine similarity to centroid) -> redundancy
removal -> important sentence selection -> original order -> structured
summary.
"""

import logging
import os
import traceback

from dotenv import load_dotenv
from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

load_dotenv()

from config import Config  # noqa: E402
from services.document_service import (  # noqa: E402
    allowed_file,
    get_file_type,
    process_document,
    save_upload,
)
from services.summarization_service import summarize_document  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("intellisum")


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

    @app.errorhandler(RequestEntityTooLarge)
    def handle_too_large(_exc):
        max_mb = app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024)
        flash(f"File too large. Maximum allowed size is {max_mb:.0f} MB.", "error")
        return redirect(url_for("index"))

    @app.errorhandler(500)
    def handle_500(_exc):
        logger.error("Internal error:\n%s", traceback.format_exc())
        flash("Something went wrong while processing your document.", "error")
        return redirect(url_for("index"))

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")

    def _render_result(doc, summary_length):
        """Run extractive NLP summarization and render the result page."""
        method = "extractive"
        try:
            result = summarize_document(doc, method=method, length=summary_length)
        except ValueError as exc:
            # User-friendly: too little text.
            flash(str(exc), "error")
            return redirect(url_for("index"))
        except Exception:
            logger.error("Summarization failed:\n%s", traceback.format_exc())
            flash("Could not generate a summary for this document.", "error")
            return redirect(url_for("index"))

        return render_template(
            "result.html",
            filename=doc["filename"],
            file_type=doc["file_type"],
            page_count=doc["page_count"],
            file_size_bytes=doc["file_size_bytes"],
            word_count=doc["word_count"],
            sentence_count=doc["sentence_count"],
            keywords=doc["keywords"],
            sections=doc["sections"],
            section_count=doc["section_count"],
            metadata=doc["metadata"],
            text_preview=doc["text"][:2000],
            summary_length=summary_length,
            method=method,
            doc_id=os.path.basename(doc["stored_path"]),
            result=result,
            stats=result["statistics"],
            document_info=result.get("document_info", {}),
        )

    @app.route("/summarize", methods=["POST"])
    def summarize():
        # --- Validate the upload exists ---
        if "file" not in request.files:
            flash("No file uploaded. Please choose a PDF, DOCX or TXT file.", "error")
            return redirect(url_for("index"))

        file = request.files["file"]
        if not file or not (file.filename or "").strip():
            flash("No file selected. Please choose a file first.", "error")
            return redirect(url_for("index"))

        summary_length = request.form.get("summary_length", "medium").lower()
        if summary_length not in {"short", "medium", "detailed"}:
            summary_length = "medium"
        # Local-only engine: summarization is always extractive NLP.
        method = "extractive"

        # --- Extension validation (never execute uploaded files) ---
        if not allowed_file(file.filename):
            flash(
                f"Unsupported file format. Supported formats: PDF, DOCX, TXT. "
                f"Got: '{file.filename}'.",
                "error",
            )
            return redirect(url_for("index"))

        # --- Save + process ---
        try:
            saved_path = save_upload(file, app.config["UPLOAD_FOLDER"])
        except ValueError as exc:
            flash(str(exc), "error")
            return redirect(url_for("index"))
        except Exception:
            logger.error("Save failed:\n%s", traceback.format_exc())
            flash("Could not save the uploaded file. Please try again.", "error")
            return redirect(url_for("index"))

        try:
            file_type = get_file_type(file.filename)
            doc = process_document(saved_path, file_type=file_type)
        except ValueError as exc:
            # User-friendly errors: empty / corrupted / no text / unsupported.
            flash(str(exc), "error")
            return redirect(url_for("index"))
        except Exception:
            logger.error("Processing failed:\n%s", traceback.format_exc())
            flash("Could not process this document. It may be corrupted.", "error")
            return redirect(url_for("index"))

        return _render_result(doc, summary_length)

    @app.route("/resummarize", methods=["POST"])
    def resummarize():
        """Re-run extractive summarization on an already-uploaded file.

        Lets the user try a different summary length without uploading again.
        """
        doc_id = secure_filename(request.form.get("doc_id", ""))
        if not doc_id:
            flash("Document reference missing. Please upload the file again.", "error")
            return redirect(url_for("index"))

        summary_length = request.form.get("summary_length", "medium").lower()
        if summary_length not in {"short", "medium", "detailed"}:
            summary_length = "medium"

        upload_dir = os.path.abspath(app.config["UPLOAD_FOLDER"])
        saved_path = os.path.abspath(os.path.join(upload_dir, doc_id))
        # Containment check: never read outside the upload directory.
        if os.path.dirname(saved_path) != upload_dir or not os.path.exists(saved_path):
            flash("Document no longer available. Please upload it again.", "error")
            return redirect(url_for("index"))

        try:
            file_type = get_file_type(saved_path)
            doc = process_document(saved_path, file_type=file_type)
        except ValueError as exc:
            flash(str(exc), "error")
            return redirect(url_for("index"))
        except Exception:
            logger.error("Re-processing failed:\n%s", traceback.format_exc())
            flash("Could not process this document.", "error")
            return redirect(url_for("index"))

        return _render_result(doc, summary_length)

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
