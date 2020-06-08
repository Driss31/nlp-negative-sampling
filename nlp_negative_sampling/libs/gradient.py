"""Function to compute gradient."""
import logging

import numpy as np
from scipy.special import expit


def gradient(theta, nEmbed, positive_pairs, negative_pairs, nb_words, nb_contexts):
    """Compute gradient at init_theta for positive and negative pairs."""
    # Initialize gradient
    grad = np.zeros(len(theta))

    # Embedding matrix of target words
    words_matrix = theta[: nEmbed * nb_words].reshape(nb_words, nEmbed)

    # Embedding matrix of contextes
    contexts_matrix = theta[nEmbed * nb_words :].reshape(nb_contexts, nEmbed)

    logging.info("Start computing gradient...")

    # Positive pairs
    logging.info("Update gradient using positive pairs...")
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
        grad[word_index * nEmbed : (word_index + 1) * nEmbed] += df_word
        grad[
            (nb_words + context_index)
            * nEmbed : (nb_words + context_index + 1)
            * nEmbed
        ] += df_context
    logging.info("Done: Gradient updated using positive pairs.")

    # Negative pairs
    logging.info("Update gradient using negative pairs...")
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
        grad[word_index * nEmbed : (word_index + 1) * nEmbed] += df_word
        grad[
            (nb_words + context_index)
            * nEmbed : (nb_words + context_index + 1)
            * nEmbed
        ] += df_context
    logging.info("Done: Gradient updated using negative pairs.")

    return grad
