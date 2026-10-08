"""IntelliSum — Intelligent Document Summarization and Structuring System.

100% local NLP summarization engine (no LLM, no API, no network):
text extraction -> preprocessing -> sentence segmentation -> TF-IDF
-> sentence scoring (cosine similarity to centroid) -> redundancy
removal -> important sentence selection -> original order -> structured
summary.
"""

import logging
import os
import time
import traceback

from dotenv import load_dotenv
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
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
from services.output_service import (  # noqa: E402
    build_docx,
    build_pdf,
    export_filename,
    safe_output_path,
)
from services.summarization_service import (  # noqa: E402
    build_download_payload,
    summarize_document,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("intellisum")

# Generated files older than this are removed opportunistically
# (best-effort; never the file that is currently being processed).
OUTPUT_MAX_AGE_SEC = 24 * 60 * 60


def _cleanup_old_files(directory, marker="_summary"):
    """Delete stale files to bound disk usage.

    Only files containing ``marker`` (or every file except ``.gitkeep``
    when ``marker`` is "") older than OUTPUT_MAX_AGE_SEC are removed.
    """
    try:
        now = time.time()
        for name in os.listdir(directory):
            if name == ".gitkeep":
                continue
            if marker and marker not in name:
                continue
            path = os.path.join(directory, name)
            try:
                if (
                    os.path.isfile(path)
                    and now - os.path.getmtime(path) > OUTPUT_MAX_AGE_SEC
                ):
                    os.remove(path)
            except OSError:
                continue
    except OSError:
        pass


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
        max_bytes = app.config["MAX_CONTENT_LENGTH"]
        return render_template(
            "index.html",
            max_upload_mb=round(max_bytes / (1024 * 1024)),
            max_upload_bytes=max_bytes,
        )

    def _render_result(doc, summary_length):
        """Run extractive NLP summarization and render the result page."""
        _cleanup_old_files(os.path.abspath(app.config["UPLOAD_FOLDER"]), marker="")
        try:
            result = summarize_document(doc, method="extractive", length=summary_length)
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
            flash(
                "Unable to read this document. Please upload a valid file.",
                "error",
            )
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

    @app.route("/download/<fmt>", methods=["GET"])
    def download(fmt):
        """Download the structured summary as PDF or DOCX.

        The export is regenerated from the stored upload with the same
        pipeline as the result page, so the webpage, PDF and DOCX always
        contain identical data. Query params: ``doc_id``, ``length``.
        """
        fmt = (fmt or "").lower()
        if fmt not in {"pdf", "docx"}:
            flash("Unsupported export format. Choose PDF or DOCX.", "error")
            return redirect(url_for("index"))

        doc_id = secure_filename(request.args.get("doc_id", ""))
        if not doc_id:
            flash("Document reference missing. Please upload the file again.", "error")
            return redirect(url_for("index"))

        summary_length = (request.args.get("length", "medium") or "medium").lower()
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
            result = summarize_document(doc, method="extractive", length=summary_length)
            payload = build_download_payload(doc, result)
        except ValueError as exc:
            flash(str(exc), "error")
            return redirect(url_for("index"))
        except Exception:
            logger.error("Export preparation failed:\n%s", traceback.format_exc())
            flash("Could not prepare the export for this document.", "error")
            return redirect(url_for("index"))

        try:
            _cleanup_old_files(os.path.abspath(app.config["OUTPUT_FOLDER"]))
            filename = export_filename(
                doc.get("filename", "document"), summary_length, fmt
            )
            dest = safe_output_path(app.config["OUTPUT_FOLDER"], filename)
            if fmt == "pdf":
                build_pdf(payload, dest)
                mimetype = "application/pdf"
            else:
                build_docx(payload, dest)
                mimetype = (
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                )
        except Exception:
            logger.error("Export build failed:\n%s", traceback.format_exc())
            flash("Could not generate the export file. Please try again.", "error")
            return redirect(url_for("index"))

        return send_file(dest, as_attachment=True, download_name=filename, mimetype=mimetype)

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
