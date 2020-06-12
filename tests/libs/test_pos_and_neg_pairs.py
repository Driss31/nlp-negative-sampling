"""Tests for generating negative and positive pairs."""
from unittest import mock

from nlp_negative_sampling.libs.pos_and_neg_pairs import (
    get_negative_pairs,
    get_positive_pairs,
)


def test_get_positive_pairs():
    """Should return positive pairs to the given word."""
    answer = get_positive_pairs(
        [["the", "cat", "cat", "red"], ["the", "red", "cat", "cat"], ["the", "red"]], 2
    )
    assert answer == (
        [
            (0, 0),
            (1, 1),
            (1, 0),
            (1, 0),
            (1, 2),
            (2, 0),
            (0, 2),
            (2, 1),
            (2, 0),
            (1, 2),
            (1, 0),
            (1, 0),
            (0, 2),
            (2, 1),
        ],
        {"the": 0, "cat": 1, "red": 2},
        {"cat": 0, "the": 1, "red": 2},
    )


@mock.patch("numpy.random.randint", lambda x: 5)
def test_get_negative_pairs():
    """Should return negatives pairs."""
    answer = get_negative_pairs(
        [
            (0, 0),
            (1, 1),
            (1, 0),
            (1, 0),
            (1, 2),
            (2, 0),
            (0, 2),
            (2, 1),
            (2, 0),
            (1, 2),
            (1, 0),
            (1, 0),
            (0, 2),
            (2, 1),
        ],
        2,
    )
    assert answer == [
        (0, 0),
        (0, 0),
        (1, 0),
        (1, 0),
        (1, 0),
        (1, 0),
        (1, 0),
        (1, 0),
        (1, 0),
        (1, 0),
        (2, 0),
        (2, 0),
        (0, 0),
        (0, 0),
        (2, 0),
        (2, 0),
        (2, 0),
        (2, 0),
        (1, 0),
        (1, 0),
        (1, 0),
        (1, 0),
        (1, 0),
        (1, 0),
        (0, 0),
        (0, 0),
        (2, 0),
        (2, 0),
    ]
