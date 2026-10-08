"""Tests for preprocessing, keyword extraction and section detection."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nlp.keyword_extraction import extract_keywords  # noqa: E402
from nlp.preprocessing import preprocess_text  # noqa: E402
from nlp.section_detection import detect_sections  # noqa: E402


SAMPLE = (
    "Natural language processing enables computers to understand human language. "
    "Machine learning models learn patterns from large text collections. "
    "TF-IDF weighting highlights important terms across documents. "
    "Cosine similarity measures the angle between sentence vectors. "
    "Extractive summarization selects the most important sentences."
)


class TestNLP(unittest.TestCase):
    def test_preprocessing(self):
        out = preprocess_text(SAMPLE)
        self.assertEqual(out["original_text"], SAMPLE)
        self.assertGreaterEqual(out["sentence_count"], 4)
        self.assertGreater(out["word_count"], 20)
        self.assertTrue(len(out["tokens"]) > 0)
        self.assertTrue(len(out["cleaned_text"]) > 0)
        # Original preserved, cleaned differs (lowercased, stopwords removed).
        self.assertNotEqual(out["original_text"], out["cleaned_text"])

    def test_preprocessing_empty(self):
        out = preprocess_text("")
        self.assertEqual(out["sentence_count"], 0)
        self.assertEqual(out["word_count"], 0)
        self.assertEqual(out["tokens"], [])

    def test_keywords(self):
        doc = (SAMPLE + " ") * 4
        kws = extract_keywords(doc, top_n=10)
        self.assertGreaterEqual(len(kws), 5)
        self.assertLessEqual(len(kws), 15)
        # No duplicates (case-insensitive).
        lowered = [k.lower() for k in kws]
        self.assertEqual(len(lowered), len(set(lowered)))

    def test_keywords_empty(self):
        self.assertEqual(extract_keywords("", top_n=10), [])

    def test_section_detection_numbered(self):
        text = (
            "1. Introduction\n\nMachine learning is a field of study. " * 3
            + "\n\n2. Methods\n\nWe apply TF-IDF weighting to rank sentences properly. " * 3
            + "\n\n3. Conclusion\n\nThe results show clear improvements overall. " * 3
        )
        sections = detect_sections(text)
        self.assertGreaterEqual(len(sections), 2)
        titles = [s["title"] for s in sections]
        self.assertTrue(any("Introduction" in t for t in titles))

    def test_section_detection_plain_text(self):
        # Plain prose with no headings -> no invented sections.
        text = (
            "Machine learning is a field of study that gives computers the ability "
            "to learn without being explicitly programmed. It has many applications "
            "in the modern world and continues to grow rapidly every single year."
        )
        self.assertEqual(detect_sections(text), [])

    def test_section_detection_docx_hints(self):
        text = (
            "Introduction\nSome intro content here that is fairly long and meaningful. "
            * 2
            + "\nMethods\nSome methods content here that is fairly long and meaningful. "
            * 2
        )
        sections = detect_sections(
            text, headings_hint=["Introduction", "Methods"]
        )
        self.assertGreaterEqual(len(sections), 2)


if __name__ == "__main__":
    unittest.main()
