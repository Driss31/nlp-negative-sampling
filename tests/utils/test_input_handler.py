"""Tests for input handlers."""
import pandas as pd

from nlp_negative_sampling.utils.input_handler import load_pairs


def test_load_pairs(tmp_path):
    """Should return a list of words and their similarity."""
    pd.DataFrame(
        columns=["word_1", "word_2", "similarity"],
        data=[["woman", "man", 0.9], ["yellow", "red", 0.8]],
    ).to_csv(tmp_path / "pairs.csv", index=False)

    answer = load_pairs(path=str(tmp_path / "pairs.csv"))

    assert answer == [["woman", "man", 0.9], ["yellow", "red", 0.8]]
