"""Placeholder for Phase 2 summarization tests.

Phase 1 only ships upload + extraction + NLP. This file exists so the
test layout matches the final project structure. It is skipped until
Phase 2 implements the summarization engine.
"""

import unittest


@unittest.skip("Phase 2 not implemented yet (extractive/abstractive summarizers).")
class TestSummarization(unittest.TestCase):
    def test_placeholder(self):
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
