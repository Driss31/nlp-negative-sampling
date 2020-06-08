"""Function to get positive and negative pairs."""
from typing import Dict, List, Tuple

import numpy as np


def get_positive_pairs(
    processed_sentences: List[List[str]], window_size: int
) -> Tuple[List[Tuple[int, int]], Dict[str, int], Dict[str, int]]:
    """Get Pairs of words that co-occur (in the window delimited by winSize): Positive examples.

    Args:
        processed_sentences: text data that have been processed.
        window_size: number of words to look at.

    Returns:
        List of positive pairs, dictionary attributing an index to each word, vocabulary.
    """
    positive_pairs = []
    words_voc = {}
    context_voc = {}
    indexer_words = 0
    indexer_context = 0

    for sentence in processed_sentences:
        for i, word in enumerate(sentence):
            if word not in words_voc:
                words_voc[word] = indexer_words
                indexer_words += 1

            for j in range(
                max(0, i - window_size // 2),
                min(i + window_size // 2 + 1, len(sentence)),
            ):
                if i != j:  # word != context
                    if sentence[j] not in context_voc:
                        context_voc[sentence[j]] = indexer_context
                        indexer_context += 1
                    positive_pairs.append((words_voc[word], context_voc[sentence[j]]))

    return positive_pairs, words_voc, context_voc


def get_negative_pairs(
    positive_pairs: List[Tuple[int, int]], negative_rate: int
) -> List[Tuple[int, int]]:
    """Get Pairs of words that don't co-occur: Negative examples. Size: negative_rate * size(positive_pairs).

    Args:
        positive_pairs: pairs of indexes that represent co-occurring words.
        negative_rate: multiplier to get number of negative pairs to generate using number of positive pairs.

    Returns:
        List of indexes of negative pairs.
    """
    negative_pairs = []
    count_positive_pairs = len(positive_pairs)

    for p_pair in positive_pairs:
        word_index = p_pair[0]  # target word
        for _ in range(negative_rate):
            pair_index = np.random.randint(
                count_positive_pairs
            )  # Random index for a pair (can be improved)
            negative_context = positive_pairs[pair_index][
                1
            ]  # Get the random pair context index
            negative_pairs.append((word_index, negative_context))

    return negative_pairs
