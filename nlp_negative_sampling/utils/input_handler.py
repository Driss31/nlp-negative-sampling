"""Function to read tests results."""
from typing import List, Union

import pandas as pd


def load_pairs(path: str) -> List[List[Union[str, float]]]:
    """Read similarity results."""
    data = pd.read_csv(filepath_or_buffer=path)
    pairs = [list(a) for a in zip(data["word_1"], data["word_2"], data["similarity"])]
    return pairs
