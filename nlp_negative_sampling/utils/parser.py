"""Parser to get input arguments."""
from argparse import ArgumentParser


def get_command_line_parser() -> ArgumentParser:
    """Standard command line parser."""
    parser = ArgumentParser()
    parser.add_argument(
        "--text_path", help="Path to the training data set.", required=True
    )
    parser.add_argument("--model_path", help="path to embedding.", required=True)
    parser.add_argument("--test", help="", action="store_true")
    parser.add_argument("--results_path", help="Path where to save results.")

    return parser
