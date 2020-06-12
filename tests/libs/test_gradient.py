"""Tests functions to compute gradient."""
import numpy as np
import pytest

from nlp_negative_sampling.libs.gradient import (
    _update_neg_gradient,
    _update_pos_grad,
    compute_gradient,
)
from nlp_negative_sampling.libs.pos_and_neg_pairs import (
    get_negative_pairs,
    get_positive_pairs,
)

EMBEDDING_DIMENSION = 5
EPSILON = 1e-5
NEGATIVE_RATE = 2


@pytest.fixture(name="words_matrix_mock", scope="module")
def mock_words_matrix():
    """Mock a list of prohibited periods."""
    return np.array(
        [
            [
                9.38386976e-06,
                3.58025040e-06,
                1.25389021e-06,
                2.26132835e-07,
                8.81851443e-06,
            ],
            [
                9.58471775e-06,
                6.41446435e-06,
                7.06252390e-06,
                6.41948680e-06,
                8.77222948e-06,
            ],
            [
                1.80683081e-06,
                7.67582037e-06,
                7.71810921e-06,
                4.00516153e-07,
                7.40446005e-06,
            ],
        ]
    )


@pytest.fixture(name="contexts_matrix_mock", scope="module")
def mock_contexts_matrix():
    """Mock a list of prohibited periods."""
    return np.array(
        [
            [
                9.41103388e-06,
                4.83504276e-06,
                9.43120184e-06,
                3.74896957e-06,
                9.65916647e-06,
            ],
            [
                1.78013817e-06,
                5.37020514e-06,
                6.25643828e-06,
                2.48336593e-06,
                1.12196641e-07,
            ],
            [
                1.56913344e-06,
                4.34300134e-07,
                5.69841884e-06,
                7.28147202e-06,
                4.94338284e-06,
            ],
        ]
    )


def test_update_pos_grad(words_matrix_mock, contexts_matrix_mock):
    """Should return an 5-ndarray representing the gradient."""
    positive_pairs, words_voc, context_voc = get_positive_pairs(
        [["the", "cat", "cat", "red"], ["the", "red", "cat", "cat"], ["the", "red"]], 2
    )
    count_words = len(words_voc.keys())
    count_contexts = len(context_voc.keys())
    answer = _update_pos_grad(
        grad=np.zeros(EMBEDDING_DIMENSION * (count_words + count_contexts)),
        words_matrix=words_matrix_mock,
        embedding_dim=EMBEDDING_DIMENSION,
        positive_pairs=positive_pairs,
        count_words=count_words,
        contexts_matrix=contexts_matrix_mock,
    )

    np.allclose(
        answer,
        np.array(
            [
                6.27465038e-06,
                2.85182151e-06,
                1.04140198e-05,
                9.15595680e-06,
                9.77296608e-06,
                2.12812703e-05,
                1.27894882e-05,
                2.76890417e-05,
                1.60210941e-05,
                2.43178141e-05,
                1.11911720e-05,
                1.02052479e-05,
                1.56876401e-05,
                6.23233550e-06,
                9.77136311e-06,
                2.56682012e-05,
                2.22948743e-05,
                2.24701021e-05,
                1.33525562e-05,
                2.93581762e-05,
                6.59918968e-06,
                1.08830525e-05,
                1.12493712e-05,
                3.61025955e-06,
                1.17905748e-05,
                1.89685875e-05,
                9.99471475e-06,
                8.31641411e-06,
                6.64561963e-06,
                1.75907439e-05,
            ]
        ),
    )


def test_update_neg_gradient(words_matrix_mock, contexts_matrix_mock):
    """Should return an 5-ndarray representing the gradient."""
    positive_pairs, words_voc, context_voc = get_positive_pairs(
        [["the", "cat", "cat", "red"], ["the", "red", "cat", "cat"], ["the", "red"]], 2
    )
    negative_pairs = get_negative_pairs(
        positive_pairs=positive_pairs, negative_rate=NEGATIVE_RATE
    )
    count_words = len(words_voc.keys())
    count_contexts = len(context_voc.keys())
    answer = _update_neg_gradient(
        grad=_update_pos_grad(
            grad=np.zeros(EMBEDDING_DIMENSION * (count_words + count_contexts)),
            words_matrix=words_matrix_mock,
            embedding_dim=EMBEDDING_DIMENSION,
            positive_pairs=positive_pairs,
            count_words=count_words,
            contexts_matrix=contexts_matrix_mock,
        ),
        words_matrix=words_matrix_mock,
        embedding_dim=EMBEDDING_DIMENSION,
        negative_pairs=negative_pairs,
        count_words=count_words,
        contexts_matrix=contexts_matrix_mock,
    )

    np.allclose(
        answer,
        np.array(
            [
                -1.42220532e-05,
                -9.72051664e-06,
                -1.44258125e-05,
                -3.22440131e-06,
                -1.20731566e-05,
                -9.62392200e-06,
                -8.65632679e-06,
                -2.23688769e-05,
                -1.89207948e-05,
                -1.48285456e-05,
                -1.49011175e-05,
                -7.46971421e-06,
                -1.69960122e-05,
                -9.26419037e-06,
                -1.69604411e-05,
                -2.15784097e-05,
                -2.00913383e-05,
                -1.69892611e-05,
                -4.14971689e-06,
                -2.87205765e-05,
                -1.42766526e-05,
                -8.20458955e-06,
                -7.68946901e-06,
                -6.53255322e-06,
                -1.31814867e-05,
                -1.53809160e-05,
                -1.48767137e-05,
                -1.73571573e-05,
                -1.29261653e-05,
                -1.68374318e-05,
            ]
        ),
    )


def test_compute_gradient(logger):
    """Should return gradient computed over positive and negative pairs."""
    positive_pairs, words_voc, context_voc = get_positive_pairs(
        [["the", "cat", "cat", "red"], ["the", "red", "cat", "cat"], ["the", "red"]], 2
    )
    negative_pairs = get_negative_pairs(
        positive_pairs=positive_pairs, negative_rate=NEGATIVE_RATE
    )
    count_words = len(words_voc.keys())
    count_contexts = len(context_voc.keys())
    answer = compute_gradient(
        logger=logger,
        theta=np.array(
            [
                9.38386976e-06,
                3.58025040e-06,
                1.25389021e-06,
                2.26132835e-07,
                8.81851443e-06,
                9.58471775e-06,
                6.41446435e-06,
                7.06252390e-06,
                6.41948680e-06,
                8.77222948e-06,
                1.80683081e-06,
                7.67582037e-06,
                7.71810921e-06,
                4.00516153e-07,
                7.40446005e-06,
                9.41103388e-06,
                4.83504276e-06,
                9.43120184e-06,
                3.74896957e-06,
                9.65916647e-06,
                1.78013817e-06,
                5.37020514e-06,
                6.25643828e-06,
                2.48336593e-06,
                1.12196641e-07,
                1.56913344e-06,
                4.34300134e-07,
                5.69841884e-06,
                7.28147202e-06,
                4.94338284e-06,
            ]
        ),
        embedding_dim=EMBEDDING_DIMENSION,
        positive_pairs=positive_pairs,
        negative_pairs=negative_pairs,
        count_words=count_words,
        count_contexts=count_contexts,
    )

    np.allclose(
        answer,
        np.array(
            [
                -1.42220532e-05,
                -9.72051664e-06,
                -1.44258125e-05,
                -3.22440131e-06,
                -1.20731566e-05,
                -9.62392200e-06,
                -8.65632679e-06,
                -2.23688769e-05,
                -1.89207948e-05,
                -1.48285456e-05,
                -1.49011175e-05,
                -7.46971421e-06,
                -1.69960122e-05,
                -9.26419037e-06,
                -1.69604411e-05,
                -2.15784097e-05,
                -2.00913383e-05,
                -1.69892611e-05,
                -4.14971689e-06,
                -2.87205765e-05,
                -1.42766526e-05,
                -8.20458955e-06,
                -7.68946901e-06,
                -6.53255322e-06,
                -1.31814867e-05,
                -1.53809160e-05,
                -1.48767137e-05,
                -1.73571573e-05,
                -1.29261653e-05,
                -1.68374318e-05,
            ]
        ),
    )
