"""Utils for preparing test data."""
from collections import Counter
from itertools import chain
import string
import typing
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


def flatten_list(list_of_lists: List[List[str]]) -> List[str]:
    """Flatten a list of lists of strings."""
    return [word for _list in list_of_lists for word in _list]


def count_words(words: List[List[str]]) -> typing.Counter[str]:
    """Return a dictionary with words frequencies."""
    return Counter(chain.from_iterable(words))


def rare_word_pruning(list_tokens: List[List[str]], min_count: int) -> List[List[str]]:
    """Remove words that occur less than min_count time in file."""
    count_tokens = count_words(words=list_tokens)
    return [
        [word for word in sent if count_tokens[word] > min_count]
        for sent in list_tokens
    ]
