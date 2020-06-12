"""Functions to compute similarity between words."""
import logging
from typing import Dict, List, Tuple

import numpy as np

OOV_EMBEDDING_VALUE = 0.01


def _get_word_embedding(
    logger: logging.Logger,
    word: str,
    words_voc: Dict[str, int],
    embed_matrix: np.ndarray,
    default_embed: np.ndarray,
) -> np.ndarray:
    """Return index of word if it's in words_voc."""
    try:
        word_idx = words_voc[word]
        word_embedding = embed_matrix[word_idx]
    except KeyError:
        logger.info("Out of Vocabulary:", word=word)
        word_embedding = default_embed

    return word_embedding


def _cosine_similarity(vect_1: np.ndarray, vect_2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return abs(vect_1.dot(vect_2) / (np.linalg.norm(vect_1) * np.linalg.norm(vect_2)))


def words_similarity(
    logger: logging.Logger,
    word_1,
    word_2,
    words_voc: Dict[str, int],
    embed_matrix: np.ndarray,
) -> float:
    """Compute cosine similarity between two words.

    Unknown words are mapped to one common vector.

    Note:
        Words that aren't in the training set are considered as "OOV".
        They get the same vector with low values.
    Args:
        logger: logger.
        word_1: first word to compare.
        word_2: second word to compare.
        words_voc: dictionary containing words and their indexes.
        embed_matrix: matrix of embeddings.

    Returns:
        The similarity between word1 and word2. A value in [0, 1] (the higher the more similar).
    """
    dim_embed = embed_matrix.shape[1]
    default_embed = np.ones(dim_embed) * OOV_EMBEDDING_VALUE

    word_1_embedding = _get_word_embedding(
        logger=logger,
        word=word_1,
        words_voc=words_voc,
        embed_matrix=embed_matrix,
        default_embed=default_embed,
    )
    word_2_embedding = _get_word_embedding(
        logger=logger,
        word=word_2,
        words_voc=words_voc,
        embed_matrix=embed_matrix,
        default_embed=default_embed,
    )

    return _cosine_similarity(vect_1=word_1_embedding, vect_2=word_2_embedding)


def find_k_most_similar(
    logger: logging.Logger,
    word_target: str,
    k: int,
    words_voc: Dict[str, int],
    embed_matrix: np.ndarray,
) -> Tuple[List[str], Dict[str, int]]:
    """Return the k most similar words to `word` and the similarity score.

    Args:
        logger: logger.
        word_target: target word.
        k: number of similar words to look for.
        words_voc: dictionary of words and their indexes.
        embed_matrix: matrix containing words embeddings.

    Returns:
        A ranked list of the most similar words and a dictionary containing their similarity scores.
    """
    similarity_dict = {
        word: words_similarity(
            logger=logger,
            word_1=word_target,
            word_2=word,
            words_voc=words_voc,
            embed_matrix=embed_matrix,
        )
        for word in words_voc
    }

    ranked_similar_words = sorted(
        similarity_dict, key=similarity_dict.get, reverse=True
    )

    return ranked_similar_words[1 : k + 1], similarity_dict
