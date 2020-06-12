"""Tests for parser."""
import unittest

from nlp_negative_sampling.utils.parser import get_command_line_parser


class ParserTest(unittest.TestCase):
    """Class to test parser."""

    def setUp(self):
        """Set up a parser."""
        self.parser = get_command_line_parser()

    def test_get_command_line_parser(self):
        """Should return a parser with given arguments."""
        parsed = self.parser.parse_args(
            ["--text_path", "text.txt", "--model_path", "train"]
        )
        self.assertEqual(parsed.text_path, "text.txt")
        self.assertEqual(parsed.model_path, "train")
