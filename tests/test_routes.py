"""Route tests: upload validation, supported/unsupported/empty/corrupt files."""

import io
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app  # noqa: E402


class TestRoutes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.app.config["UPLOAD_FOLDER"] = self.tmp.name
        self.client = self.app.test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def _post(self, filename, content):
        return self.client.post(
            "/summarize",
            data={"file": (io.BytesIO(content), filename)},
            content_type="multipart/form-data",
        )

    def test_index_loads(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"IntelliSum", resp.data)
        self.assertIn(b"Summary length", resp.data)
        self.assertIn(b"Extractive NLP", resp.data)
        self.assertNotIn(b"Abstractive AI", resp.data)

    def test_txt_upload(self):
        text = (
            "Natural language processing enables computers to understand text. "
            "Machine learning learns patterns from large collections of documents. "
            "TF-IDF highlights the most important terms in each sentence. "
            "Cosine similarity compares sentence vectors for ranking purposes. "
            "Extractive methods select the most representative sentences directly."
        )
        resp = self._post("sample.txt", text.encode("utf-8"))
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Document Summary", resp.data)
        self.assertIn(b"Extractive NLP", resp.data)
        self.assertNotIn(b"Abstractive AI", resp.data)

    def test_docx_upload(self):
        from docx import Document

        doc = Document()
        doc.add_heading("Introduction", level=1)
        doc.add_paragraph(
            "Natural language processing enables computers to understand text. " * 4
        )
        doc.add_heading("Methods", level=1)
        doc.add_paragraph("TF-IDF weighting highlights important terms. " * 4)
        buf = io.BytesIO()
        doc.save(buf)
        resp = self.client.post(
            "/summarize",
            data={"file": (io.BytesIO(buf.getvalue()), "sample.docx")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Document Summary", resp.data)

    def test_pdf_upload(self):
        try:
            import pymupdf
        except ImportError:
            import fitz as pymupdf

        doc = pymupdf.open()
        page = doc.new_page()
        rect = pymupdf.Rect(72, 72, 500, 750)
        page.insert_textbox(
            rect,
            "Machine learning enables computers to learn from data. " * 10,
        )
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()
        resp = self.client.post(
            "/summarize",
            data={"file": (io.BytesIO(buf.getvalue()), "sample.pdf")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Document Summary", resp.data)

    def test_invalid_extension(self):
        resp = self._post("evil.exe", b"malicious content")
        self.assertEqual(resp.status_code, 302)  # redirect with flash message

    def test_empty_file(self):
        resp = self._post("empty.txt", b"")
        self.assertEqual(resp.status_code, 302)

    def test_corrupted_pdf(self):
        resp = self._post("broken.pdf", b"this is definitely not a pdf")
        self.assertIn(resp.status_code, (200, 302))

    def test_no_file_part(self):
        resp = self.client.post("/summarize", data={})
        self.assertEqual(resp.status_code, 302)


TEXT_12 = (
    "Natural language processing enables computers to understand human language. "
    "Machine learning models learn useful patterns from large text collections. "
    "TF-IDF weighting highlights terms that distinguish one sentence from others. "
    "Cosine similarity measures the angle between two sentence vectors. "
    "Neural networks capture complex nonlinear relationships in language data. "
    "Text classification assigns predefined labels to entire documents. "
    "Named entity recognition locates people and places mentioned in text. "
    "Sentiment analysis determines whether expressed opinions are positive. "
    "Machine translation converts sentences from one language into another. "
    "Topic modelling discovers hidden themes across a document collection. "
    "Word embeddings represent vocabulary items as dense numeric vectors. "
    "Evaluation metrics such as precision and recall quantify model quality."
)


class TestPhase2Routes(unittest.TestCase):
    """Extractive-only routes, lengths, resummarize (local NLP)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.app.config["UPLOAD_FOLDER"] = self.tmp.name
        self.client = self.app.test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def _upload(self, length="medium"):
        return self.client.post(
            "/summarize",
            data={
                "file": (io.BytesIO(TEXT_12.encode("utf-8")), "doc.txt"),
                "summary_length": length,
            },
            content_type="multipart/form-data",
        )

    def test_extractive_all_lengths(self):
        for length in ("short", "medium", "detailed"):
            resp = self._upload(length=length)
            self.assertEqual(resp.status_code, 200, f"length={length}")
            for section in (
                b"Executive Summary", b"Key Points",
                b"Important Keywords", b"Summary Statistics",
                b"Selected Important Sentences",
            ):
                self.assertIn(section, resp.data, f"{length}: {section}")
            self.assertIn(b"Extractive NLP", resp.data)
            self.assertNotIn(b"Abstractive AI", resp.data)

    def test_too_little_text_message(self):
        resp = self.client.post(
            "/summarize",
            data={"file": (io.BytesIO(b"Hi there friend."), "tiny.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 302)  # redirect + flash message

    def test_resummarize_flow(self):
        first = self._upload(length="short")
        self.assertEqual(first.status_code, 200)
        saved = os.listdir(self.tmp.name)
        self.assertTrue(saved)
        resp = self.client.post(
            "/resummarize",
            data={
                "doc_id": saved[0],
                "summary_length": "detailed",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Executive Summary", resp.data)

    def test_resummarize_invalid_doc(self):
        resp = self.client.post(
            "/resummarize",
            data={"doc_id": "nope.txt", "summary_length": "short"},
        )
        self.assertEqual(resp.status_code, 302)


if __name__ == "__main__":
    unittest.main()
