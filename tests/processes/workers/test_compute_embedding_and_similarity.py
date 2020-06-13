"""Test for the worker."""
import os.path

import pandas as pd

from nlp_negative_sampling.processes.workers.compute_embedding_and_similarity import (
    skip_gram_similarity_worker,
)
from nlp_negative_sampling.utils.parser import get_command_line_parser


def test_skip_gram_similarity_worker_train():
    """Should train and save the model."""
    parser = get_command_line_parser()
    skip_gram_similarity_worker(
        parser.parse_args(
            [
                "--text_path",
                "data/test_worker/text.txt",
                "--model_path",
                "data/test_worker/trained_model/",
            ]
        )
    )

    os.path.isfile("data/test_worker/trained_model/embed_matrix")
    os.path.isfile("data/test_worker/trained_model/words_voc")


def test_skip_gram_similarity_worker_predict():
    """Should compute similarity between words of vocabulary."""
    parser = get_command_line_parser()
    skip_gram_similarity_worker(
        parser.parse_args(
            [
                "--text_path",
                "data/test_worker/test_results/pairs.csv",
                "--model_path",
                "data/test_worker/trained_model/",
                "--test",
                "--results_path",
                "data/test_worker/test_results/results.csv",
            ]
        )
    )

    os.path.isfile("data/test_worker/test_results/results.csv")
    assert list(pd.read_csv("data/test_worker/test_results/results.csv").columns) == [
        "word_1",
        "word_2",
        "similarity",
    ]
