"""Function to read tests results."""
from typing import Iterator, Tuple

import pandas as pd


def load_pairs(path: str) -> Iterator[Tuple[str, str, float]]:
    """Read results."""
    data = pd.read_csv(filepath_or_buffer=path, delimiter="\t")
    pairs = zip(data["word_1"], data["word_2"], data["similarity"])
    return pairs
