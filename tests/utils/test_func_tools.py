"""Tests for func_tools."""
from nlp_negative_sampling.utils.func_tools import compose


def test_compose():
    """Should compose the 3 given functions."""
    pipeline = compose(lambda x: x + 1, lambda x: 2 * x, lambda x: x ** 2 + 3)
    assert pipeline(3) == 25
