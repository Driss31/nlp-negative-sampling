"""Utils for preparing test data."""
import string
from typing import List

from nlp_negative_sampling.utils.func_tools import compose


def get_list_lower_words(sent: str) -> List[str]:
    """Separates words in sentences and return their lower value."""
    return sent.lower().split()


def remove_punctuation(sent: List[str]) -> List[str]:
    """Remove punctuation from list of strings."""
    punctuation_remover = str.maketrans("", "", string.punctuation)
    return [
        word.translate(punctuation_remover)
        for word in sent
        if word not in string.punctuation
    ]


def keep_alphabetical_words(sent: List[str]) -> List[str]:
    """Keep only alphabetical words."""
    return [word for word in sent if word.isalpha()]


def tokenize_file(file: str) -> List[List[str]]:
    """Transform text to list of lists of words."""
    sentences = [line.rstrip("\n") for line in open(file, encoding="utf8")]
    pipeline = compose(
        keep_alphabetical_words, remove_punctuation, get_list_lower_words,
    )  # type: ignore
    return [pipeline(sent) for sent in sentences]
