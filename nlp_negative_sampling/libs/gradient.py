"""Function to compute gradient."""
import logging
from typing import List, Tuple

import numpy as np
from scipy.special import expit


def _update_pos_grad(
    grad: np.ndarray,
    words_matrix: np.ndarray,
    embedding_dim: int,
    positive_pairs: List[Tuple[int, int]],
    count_words: int,
    contexts_matrix: np.ndarray,
) -> np.ndarray:
    """Update gradient using positive pairs."""
    for p_pair in positive_pairs:
        # Get indexes of (word, context)
        word_index = p_pair[0]
        context_index = p_pair[1]

        # Get the actual embedding of the word and its context
        word = words_matrix[word_index]
        context = contexts_matrix[context_index]

        # We compute the derivative of the formula given by 'Yoav Goldberg' and 'Omer Levy'
        df_word = context * expit(-word.dot(context))
        df_context = word * expit(-word.dot(context))

        # We actualize the gradient of the word and its context
        grad[word_index * embedding_dim : (word_index + 1) * embedding_dim] += df_word
        grad[
            (count_words + context_index)
            * embedding_dim : (count_words + context_index + 1)
            * embedding_dim
        ] += df_context

    return grad


def _update_neg_gradient(
    grad: np.ndarray,
    words_matrix: np.ndarray,
    embedding_dim: int,
    negative_pairs: List[Tuple[int, int]],
    count_words: int,
    contexts_matrix: np.ndarray,
) -> np.ndarray:
    """Update gradient using negative pairs."""
    for n_pair in negative_pairs:
        # Get indexes of (word, negative context)
        word_index = n_pair[0]
        context_index = n_pair[1]

        # Get the actual embedding of the word and its context
        word = words_matrix[word_index]
        context = contexts_matrix[context_index]

        # We compute the derivative of the formula given by 'Yoav Goldberg' and 'Omer Levy'
        df_word = -context * expit(word.dot(context))
        df_context = -word * expit(word.dot(context))

        # We actualize the gradient of the word and its negative context
        grad[word_index * embedding_dim : (word_index + 1) * embedding_dim] += df_word
        grad[
            (count_words + context_index)
            * embedding_dim : (count_words + context_index + 1)
            * embedding_dim
        ] += df_context
    return grad


def compute_gradient(
    logger: logging.Logger,
    theta: np.ndarray,
    embedding_dim: int,
    positive_pairs: List[Tuple[int, int]],
    negative_pairs: List[Tuple[int, int]],
    count_words: int,
    count_contexts: int,
):
    """Compute gradient at init_theta for positive and negative pairs."""
    # Initialize gradient
    grad = np.zeros(len(theta))

    # Embedding matrix of target words
    words_matrix = theta[: embedding_dim * count_words].reshape(
        count_words, embedding_dim
    )

    # Embedding matrix of contexts
    contexts_matrix = theta[embedding_dim * count_words :].reshape(
        count_contexts, embedding_dim
    )

    logger.info("Start computing gradient...")

    # Positive pairs
    grad = _update_pos_grad(
        grad=grad,
        words_matrix=words_matrix,
        embedding_dim=embedding_dim,
        positive_pairs=positive_pairs,
        count_words=count_words,
        contexts_matrix=contexts_matrix,
    )
    logger.info("Gradient updated using positive pairs.")

    # Negative pairs
    grad = _update_neg_gradient(
        grad=grad,
        words_matrix=words_matrix,
        embedding_dim=embedding_dim,
        negative_pairs=negative_pairs,
        count_words=count_words,
        contexts_matrix=contexts_matrix,
    )
    logger.info("Gradient updated using negative pairs.")

    return grad
