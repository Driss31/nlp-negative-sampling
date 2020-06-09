"""Worker to compute similarity between words using Skip Gram model."""
from argparse import Namespace
import logging

import pandas as pd

from nlp_negative_sampling.libs.similarity import words_similarity
from nlp_negative_sampling.libs.skip_gram import SkipGram
from nlp_negative_sampling.utils.input_handler import load_pairs
from nlp_negative_sampling.utils.parser import get_command_line_parser
from nlp_negative_sampling.utils.process_text_data import tokenize_file

logger = logging.getLogger()


def skip_gram_similarity_worker(args: Namespace) -> None:
    """Train, save and test Skip Gram model."""
    if not args.test:
        sentences = tokenize_file(file=args.text_path)
        sg_model = SkipGram(logger=logger, sentences=sentences)
        sg_model.train()
        sg_model.save_model(pickle_path=args.model_path)
    else:
        pairs = load_pairs(args.text_path)
        embed_matrix, words_voc = SkipGram.load_model(args.model_path)

        results_df = pd.DataFrame(
            columns=["word_1", "word_2", "similarity"],
            data=[
                [
                    word_1,
                    word_2,
                    words_similarity(
                        logger=logger,
                        word_1=word_1,
                        word_2=word_2,
                        words_voc=words_voc,
                        embed_matrix=embed_matrix,
                    ),
                ]
                for word_1, word_2, _ in pairs
            ],
        )
        results_df.to_csv(path_or_buf=args.results_path, index=False)


if __name__ == "__main__":
    skip_gram_similarity_worker(args=get_command_line_parser().parse_args())
