"""Skip Gram object."""
import logging
import pickle
from typing import Dict, List, Tuple

import numpy as np
import tqdm

from nlp_negative_sampling.libs.gradient import compute_gradient
from nlp_negative_sampling.libs.pos_and_neg_pairs import (
    get_negative_pairs,
    get_positive_pairs,
)
from nlp_negative_sampling.utils.process_text_data import rare_word_pruning

BATCH_SIZE = 500
EMBEDDING_DIMENSION = 100
EPOCHS = 5
EPSILON = 1e-5
LEARNING_RATE = 0.01
NEGATIVE_RATE = 5
MIN_COUNT = 5
WINDOW_SIZE = 7


class SkipGram:
    """Skip Gram Model."""

    def __init__(
        self, logger: logging.Logger, sentences: List[List[str]],
    ):
        """Instantiate SkipGram model."""
        self._logger = logger
        self._logger.info("Start Initialization.")

        self.processed_sentences = rare_word_pruning(
            list_tokens=sentences, min_count=MIN_COUNT
        )
        self._logger.info(
            "Data processing ended.", count_sentences=len(self.processed_sentences)
        )

        self.positive_pairs, self.words_voc, self.context_voc = get_positive_pairs(
            processed_sentences=self.processed_sentences, window_size=WINDOW_SIZE
        )
        self._logger.info("Positive pairs generated.")

        self.negative_pairs = get_negative_pairs(
            positive_pairs=self.positive_pairs, negative_rate=NEGATIVE_RATE
        )
        self._logger.info("Negative pairs generated.")

        self._logger.info("End of Initialization.")

    def train(self) -> None:
        """Create embedding matrix."""
        count_words = len(self.words_voc.keys())
        count_contexts = len(self.context_voc.keys())
        count_pos_pairs = len(self.positive_pairs)

        # Initialize theta: vector of parameters
        count_parameters = EMBEDDING_DIMENSION * (count_words + count_contexts)
        theta = np.random.random(count_parameters) * EPSILON

        # Compute Stochastic Gradient
        self._logger.info(
            "Start Training",
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
        )
        for epoch in range(EPOCHS):
            self._logger.info(f"Epoch {epoch + 1} / {EPOCHS}")

            for batch_number in tqdm.tqdm(range(count_pos_pairs // BATCH_SIZE)):
                logging.info(
                    f"batch_number {batch_number + 1} / {count_pos_pairs // BATCH_SIZE}"
                )
                batch_begin = batch_number * BATCH_SIZE
                batch_end = min((batch_number + 1) * BATCH_SIZE, count_pos_pairs)
                batch_positive = self.positive_pairs[batch_begin:batch_end]
                batch_negative = self.negative_pairs[
                    NEGATIVE_RATE * batch_begin : NEGATIVE_RATE * batch_end
                ]

                # Compute the gradient at theta
                grad = compute_gradient(
                    theta=theta,
                    embedding_dim=EMBEDDING_DIMENSION,
                    positive_pairs=batch_positive,
                    negative_pairs=batch_negative,
                    count_words=count_words,
                    count_contexts=count_contexts,
                )

                # Actualize theta (since we want to maximize the 'loss', we add grad)
                theta = theta + LEARNING_RATE * grad

        # Matrix of embeddings
        self.embed_matrix = theta[: EMBEDDING_DIMENSION * count_words].reshape(
            count_words, EMBEDDING_DIMENSION
        )

    def save_model(self, pickle_path: str) -> None:
        """Save words and their embeddings in a pickle file."""
        with open(pickle_path + "\\embed_matrix", "wb") as file:
            mon_pickler = pickle.Pickler(file)
            mon_pickler.dump(self.embed_matrix)
        with open(pickle_path + "\\words_voc", "wb") as file:
            mon_pickler = pickle.Pickler(file)
            mon_pickler.dump(self.words_voc)

    @staticmethod
    def load_model(pickle_path: str) -> Tuple[np.ndarray, Dict[str, int]]:
        """Load Skip Gram model."""
        with open(pickle_path + "\\embed_matrix", "rb") as file:
            my_depickler = pickle.Unpickler(file)
            embed_matrix = my_depickler.load()
        with open(pickle_path + "\\words_voc", "rb") as file:
            my_depickler = pickle.Unpickler(file)
            words_voc = my_depickler.load()

        return embed_matrix, words_voc
