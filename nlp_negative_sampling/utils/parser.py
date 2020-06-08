"""Parser to get input arguments."""
from argparse import ArgumentParser


def get_command_line_parser() -> ArgumentParser:
    """Standard command line parser."""
    parser = ArgumentParser()
    parser.add_argument("--text", help="Path to the training data set", required=True)
    parser.add_argument(
        "--model", help="path to embedding (W and words_voc)", required=True
    )
    parser.add_argument("--test", help="", action="store_true")

    return parser
