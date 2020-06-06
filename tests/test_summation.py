"""File for testing."""
from nlp_negative_sampling.summation import summ


def test_summation():
    """Should return the sum of two numbers."""
    answer = summ(3, 4)
    assert answer == 7
