"""Tests for similarity functions."""
import unittest.mock as mock

import numpy as np
from numpy.testing import assert_approx_equal
import pytest

from nlp_negative_sampling.libs.similarity import (
    _cosine_similarity,
    _get_word_embedding,
    find_k_most_similar,
    words_similarity,
)


@pytest.fixture
def logger(mocker) -> mock.Mock:
    """Mock a logger client."""
    return mocker.Mock()


def test_get_word_embedding_existing(logger):
    """Should return the embedding of an existing word."""
    answer = _get_word_embedding(
        logger=logger,
        word="red",
        words_voc={"black": 0, "yellow": 1, "red": 2, "white": 3},
        embed_matrix=np.array(
            [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]]
        ),
        default_embed=np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
    )
    assert np.array_equal(answer, [3, 3, 3, 3, 3])


def test_get_word_embedding_missing(logger):
    """Should return the embedding of an OOV."""
    answer = _get_word_embedding(
        logger=logger,
        word="blue",
        words_voc={"black": 0, "yellow": 1, "red": 2, "white": 3},
        embed_matrix=np.array(
            [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]]
        ),
        default_embed=np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
    )
    assert np.array_equal(answer, [0.1, 0.1, 0.1, 0.1, 0.1])


def test_cosine_similarity():
    """Should return the cosine distance approximated to 6 numbers."""
    answer = _cosine_similarity(
        vect_1=np.array([1, 1, 1, 1, 1]), vect_2=np.array([1, 0, 1, 2, 0])
    )
    assert_approx_equal(actual=answer, desired=0.730296, significant=6, verbose=False)


def test_words_similarity(logger):
    """Should return similarity between red and black."""
    answer = words_similarity(
        logger=logger,
        word_1="red",
        word_2="black",
        words_voc={"black": 0, "yellow": 1, "red": 2, "white": 3},
        embed_matrix=np.array(
            [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [1, 0, 1, 2, 0], [4, 4, 4, 4, 4]]
        ),
    )

    assert_approx_equal(actual=answer, desired=0.730296, significant=6, verbose=False)


def test_find_k_most_similar(logger):
    """Should return green and white as most similar words to red."""
    answer = find_k_most_similar(
        logger=logger,
        word_target="red",
        k=2,
        words_voc={
            "black": 0,
            "yellow": 1,
            "red": 2,
            "white": 3,
            "blue": 4,
            "green": 5,
        },
        embed_matrix=np.array(
            [
                [1, 3, 2, 6, 8],
                [3, 2, 9, 12, 0.3],
                [21, 0.87, 8, 3, 3],
                [4, 9, 2, 4, 4],
                [1, 3, 2, 4, 4],
                [4, 9, 8, 3, 3],
            ]
        ),
    )

    assert answer == (
        ["green", "white"],
        {
            "black": 0.33398431659742517,
            "yellow": 0.49171597833790753,
            "red": 1.0,
            "white": 0.4994862678676181,
            "blue": 0.4098091029624984,
            "green": 0.5677188782302137,
        },
    )
