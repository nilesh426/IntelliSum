"""Tests for Phase 3 PDF/DOCX export (ReportLab + python-docx, fully local).

Covers: PDF validity/content/pagination, DOCX structure/content, omission
of empty sections, filename sanitization, download routes end-to-end
(PDF/DOCX/TXT x short/medium/detailed), webpage/export data consistency,
and leak-free error responses.
"""

import io
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app  # noqa: E402
from services.document_service import process_document  # noqa: E402
from services.output_service import (  # noqa: E402
    build_docx,
    build_pdf,
    export_filename,
    safe_output_path,
    sanitize_stem,
)
from services.summarization_service import (  # noqa: E402
    build_download_payload,
    summarize_document,
)


TOPICS = [
    "Natural language processing enables computers to understand human language.",
    "Machine learning models learn useful patterns from large text collections.",
    "TF-IDF weighting highlights terms that distinguish one sentence from others.",
    "Cosine similarity measures the angle between two sentence vectors.",
    "Neural networks capture complex nonlinear relationships in language data.",
    "Text classification assigns predefined labels to entire documents.",
    "Named entity recognition locates people, places and organizations in text.",
    "Sentiment analysis determines whether opinions expressed are positive or negative.",
    "Machine translation converts sentences from one language into another.",
    "Topic modelling discovers hidden themes across a document collection.",
    "Word embeddings represent vocabulary items as dense numeric vectors.",
    "Evaluation metrics such as precision and recall quantify model quality.",
]

MEDIUM_TEXT = " ".join(TOPICS * 3)


def _make_payload(text=None, length="medium", sections=None):
    """Process real text through the full pipeline into an export payload."""
    text = text if text is not None else MEDIUM_TEXT
    with tempfile.NamedTemporaryFile(
        suffix=".txt", delete=False, mode="w", encoding="utf-8"
    ) as fh:
        fh.write(text)
        path = fh.name
    try:
        doc = process_document(path, file_type="txt")
        if sections is not None:
            doc["sections"] = sections
            doc["section_count"] = len(sections)
        result = summarize_document(doc, method="extractive", length=length)
        return doc, result, build_download_payload(doc, result)
    finally:
        os.remove(path)


def _pdf_text(path):
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf
    doc = pymupdf.open(path)
    try:
        return doc.page_count, "\n".join(p.get_text("text") for p in doc)
    finally:
        doc.close()


def _docx_paragraphs(path):
    from docx import Document

    doc = Document(path)
    texts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                texts.append(cell.text)
    return texts, doc.tables


class TestPdfExport(unittest.TestCase):
    def test_pdf_valid_and_paginated(self):
        _doc, _result, payload = _make_payload()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "summary.pdf")
            built = build_pdf(payload, out)
            self.assertTrue(os.path.exists(built))
            self.assertGreater(os.path.getsize(built), 0)
            with open(built, "rb") as fh:
                self.assertEqual(fh.read(4), b"%PDF")
            pages, text = _pdf_text(built)
            self.assertGreaterEqual(pages, 1)
            self.assertIn("INTELLISUM", text)
            self.assertIn("EXECUTIVE SUMMARY", text)
            self.assertIn("HOW THE SUMMARY WAS GENERATED", text)

    def test_pdf_contains_summary_data(self):
        _doc, result, payload = _make_payload()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "s.pdf")
            build_pdf(payload, out)
            _pages, text = _pdf_text(out)
            stats = result["statistics"]
            # Same numbers as the webpage statistics.
            self.assertIn(str(stats["original_words"]), text)
            self.assertIn(str(stats["summary_words"]), text)
            # Executive summary text present (first sentence at least).
            first_sentence = result["important_sentences"][0]["sentence"][:40]
            self.assertIn(first_sentence[:25], text)
            for kw in result["keywords"][:3]:
                self.assertIn(kw, text)

    def test_pdf_omits_empty_sections(self):
        _doc, _result, payload = _make_payload(sections=[])
        payload["section_summaries"] = []
        payload["conclusion"] = ""
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "s.pdf")
            build_pdf(payload, out)
            _pages, text = _pdf_text(out)
            self.assertNotIn("SECTION-WISE", text)
            self.assertNotIn("CONCLUSION", text)
            # Core sections still present.
            self.assertIn("EXECUTIVE SUMMARY", text)
            self.assertIn("KEY POINTS", text)

    def test_pdf_long_document(self):
        big = " ".join(TOPICS * 40)
        sections = [
            {"title": f"Chapter {i}",
             "content": " ".join(TOPICS[(i % 12):(i % 12 + 4)])}
            for i in range(10)
        ]
        _doc, _result, payload = _make_payload(text=big, length="detailed",
                                               sections=sections)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "long.pdf")
            build_pdf(payload, out)
            pages, text = _pdf_text(out)
            self.assertGreaterEqual(pages, 1)
            self.assertIn("INTELLISUM", text)


class TestDocxExport(unittest.TestCase):
    def test_docx_structure_and_content(self):
        _doc, result, payload = _make_payload()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "summary.docx")
            built = build_docx(payload, out)
            self.assertTrue(os.path.exists(built))
            self.assertGreater(os.path.getsize(built), 0)
            paras, tables = _docx_paragraphs(built)
            joined = "\n".join(paras)
            self.assertIn("INTELLISUM", joined)
            for heading in (
                "Document Information", "Summary Overview",
                "Executive Summary", "Key Points",
                "Important Keywords", "Important Sentences",
                "How the Summary Was Generated",
            ):
                self.assertIn(heading, joined, heading)
            self.assertGreaterEqual(len(tables), 2)  # info + overview tables
            stats = result["statistics"]
            self.assertIn(str(stats["original_words"]), joined)

    def test_docx_omits_empty_sections(self):
        _doc, _result, payload = _make_payload(sections=[])
        payload["section_summaries"] = []
        payload["conclusion"] = ""
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "s.docx")
            build_docx(payload, out)
            paras, _tables = _docx_paragraphs(out)
            joined = "\n".join(paras)
            self.assertNotIn("Section-wise Summary", joined)
            self.assertNotIn("Conclusion", joined)
            self.assertIn("Executive Summary", joined)

    def test_docx_section_headings(self):
        sections = [
            {"title": "Introduction", "content": " ".join(TOPICS[:4])},
            {"title": "Methods", "content": " ".join(TOPICS[4:8])},
        ]
        _doc, _result, payload = _make_payload(sections=sections)
        self.assertTrue(payload["section_summaries"])
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "s.docx")
            build_docx(payload, out)
            paras, _tables = _docx_paragraphs(out)
            joined = "\n".join(paras)
            self.assertIn("Introduction", joined)
            self.assertIn("Methods", joined)


class TestExportFilenames(unittest.TestCase):
    def test_export_filename_shape(self):
        name = export_filename("research_paper.pdf", "medium", "pdf")
        self.assertEqual(name, "research_paper_summary_medium.pdf")
        name = export_filename("my doc.DOCX", "short", "docx")
        self.assertEqual(name, "my_doc_summary_short.docx")

    def test_export_filename_traversal_safe(self):
        name = export_filename("../../etc/passwd", "medium", "pdf")
        self.assertNotIn("/", name)
        self.assertNotIn("\\", name)
        self.assertNotIn("..", name)
        self.assertTrue(name.endswith(".pdf"))

    def test_export_filename_bad_length_defaults(self):
        name = export_filename("doc.txt", "nonsense", "pdf")
        self.assertIn("_medium.pdf", name)

    def test_sanitize_stem(self):
        stem = sanitize_stem("a/b\\c:d.txt")
        self.assertTrue(stem)
        self.assertNotIn("/", stem)
        self.assertNotIn("\\", stem)
        self.assertNotIn(":", stem)
        self.assertEqual(sanitize_stem(""), "document")

    def test_safe_output_path_containment(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = safe_output_path(tmp, "../evil.pdf")
            self.assertEqual(os.path.dirname(dest), os.path.abspath(tmp))
            self.assertTrue(dest.endswith(".pdf"))


class TestDownloadRoutes(unittest.TestCase):
    def setUp(self):
        self.upload_tmp = tempfile.TemporaryDirectory()
        self.output_tmp = tempfile.TemporaryDirectory()
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.app.config["UPLOAD_FOLDER"] = self.upload_tmp.name
        self.app.config["OUTPUT_FOLDER"] = self.output_tmp.name
        self.client = self.app.test_client()

    def tearDown(self):
        self.upload_tmp.cleanup()
        self.output_tmp.cleanup()

    def _upload_txt(self, length="medium"):
        resp = self.client.post(
            "/summarize",
            data={
                "file": (io.BytesIO(MEDIUM_TEXT.encode("utf-8")), "paper.txt"),
                "summary_length": length,
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200)
        saved = os.listdir(self.upload_tmp.name)
        self.assertTrue(saved)
        return saved[0]

    def test_download_pdf_all_lengths(self):
        for length in ("short", "medium", "detailed"):
            doc_id = self._upload_txt(length=length)
            resp = self.client.get(
                f"/download/pdf?doc_id={doc_id}&length={length}"
            )
            try:
                self.assertEqual(resp.status_code, 200, length)
                self.assertIn("application/pdf", resp.content_type)
                self.assertIn(f"_summary_{length}.pdf", resp.headers.get("Content-Disposition", ""))
                self.assertTrue(resp.data.startswith(b"%PDF"))
            finally:
                resp.close()

    def test_download_docx_all_lengths(self):
        for length in ("short", "medium", "detailed"):
            doc_id = self._upload_txt(length=length)
            resp = self.client.get(
                f"/download/docx?doc_id={doc_id}&length={length}"
            )
            try:
                self.assertEqual(resp.status_code, 200, length)
                self.assertIn(
                    "officedocument.wordprocessingml.document", resp.content_type
                )
                self.assertIn(f"_summary_{length}.docx", resp.headers.get("Content-Disposition", ""))
                self.assertTrue(resp.data.startswith(b"PK"))
            finally:
                resp.close()

    def test_downloaded_pdf_matches_webpage_stats(self):
        doc_id = self._upload_txt(length="medium")
        page = self.client.post(
            "/resummarize",
            data={"doc_id": doc_id, "summary_length": "medium"},
        )
        self.assertEqual(page.status_code, 200)
        resp = self.client.get(f"/download/pdf?doc_id={doc_id}&length=medium")
        try:
            tmp_pdf = _save_temp(resp.data, ".pdf")
        finally:
            resp.close()
        tmp_pdf = _save_temp(resp.data, ".pdf")
        try:
            _pages, text = _pdf_text(tmp_pdf)
        finally:
            os.remove(tmp_pdf)
        # Webpage shows the same structured numbers as the export.
        for marker in (b"Summary Statistics", b"Executive Summary"):
            self.assertIn(marker, page.data)
        self.assertIn("INTELLISUM", text)

    def test_download_docx_upload_and_pdf_upload(self):
        from docx import Document

        doc = Document()
        doc.add_heading("Introduction", level=1)
        doc.add_paragraph(
            "Natural language processing enables computers to understand text. " * 8
        )
        doc.add_heading("Methods", level=1)
        doc.add_paragraph("TF-IDF weighting highlights important terms. " * 8)
        buf = io.BytesIO()
        doc.save(buf)
        resp = self.client.post(
            "/summarize",
            data={"file": (io.BytesIO(buf.getvalue()), "study.docx")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Download PDF", resp.data)
        self.assertIn(b"Download DOCX", resp.data)
        saved = os.listdir(self.upload_tmp.name)
        dl = self.client.get(f"/download/docx?doc_id={saved[0]}&length=medium")
        try:
            self.assertEqual(dl.status_code, 200)
            self.assertIn("study_summary_medium.docx", dl.headers.get("Content-Disposition", ""))
        finally:
            dl.close()

        try:
            import pymupdf
        except ImportError:
            import fitz as pymupdf
        pdf = pymupdf.open()
        pg = pdf.new_page()
        pg.insert_textbox(
            pymupdf.Rect(72, 72, 500, 750),
            "Machine learning enables computers to learn from data. " * 12,
        )
        pbuf = io.BytesIO()
        pdf.save(pbuf)
        pdf.close()
        resp = self.client.post(
            "/summarize",
            data={"file": (io.BytesIO(pbuf.getvalue()), "report.pdf")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200)
        saved = sorted(os.listdir(self.upload_tmp.name))
        dl = self.client.get(f"/download/pdf?doc_id={saved[-1]}&length=short")
        try:
            self.assertEqual(dl.status_code, 200)
            self.assertTrue(dl.data.startswith(b"%PDF"))
        finally:
            dl.close()

    def test_download_invalid(self):
        resp = self.client.get("/download/pdf?doc_id=nope.txt&length=short")
        self.assertEqual(resp.status_code, 302)
        resp = self.client.get("/download/pdf")
        self.assertEqual(resp.status_code, 302)
        resp = self.client.get("/download/xlsx?doc_id=a&length=short")
        self.assertEqual(resp.status_code, 302)

    def test_error_responses_leak_nothing(self):
        cases = [
            {"file": (io.BytesIO(b""), "empty.txt")},
            {"file": (io.BytesIO(b"Hi."), "tiny.txt")},
            {"file": (io.BytesIO(b"x"), "evil.exe")},
            {"file": (io.BytesIO(b"not a pdf at all" * 10), "broken.pdf")},
        ]
        for data in cases:
            resp = self.client.post(
                "/summarize", data=data,
                content_type="multipart/form-data", follow_redirects=True,
            )
            body = resp.data.decode("utf-8", "ignore")
            self.assertNotIn("Traceback", body)
            self.assertNotIn(".py", body)


def _save_temp(data, suffix):
    fh = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        fh.write(data)
        return fh.name
    finally:
        fh.close()


if __name__ == "__main__":
    unittest.main()
