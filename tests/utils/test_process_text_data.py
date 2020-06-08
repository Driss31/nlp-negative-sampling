"""Tests for text processing."""
from collections import Counter

from nlp_negative_sampling.utils.process_text_data import (
    count_words,
    flatten_list,
    get_list_lower_words,
    keep_alphabetical_words,
    remove_punctuation,
    tokenize_file,
)


def test_get_list_lower_words():
    """Should return a list of lower words."""
    answer = get_list_lower_words(
        sent="The U.S. Centers for Disease Control and Prevention initially advised school"
    )
    assert answer == [
        "the",
        "u.s.",
        "centers",
        "for",
        "disease",
        "control",
        "and",
        "prevention",
        "initially",
        "advised",
        "school",
    ]


def test_remove_punctuation():
    """Should remove punctuation characters."""
    answer = remove_punctuation(
        sent=[
            "the",
            "u.s.",
            "centers",
            "for",
            "disease",
            "control",
            "and",
            "prevention",
            "!",
        ]
    )
    assert answer == [
        "the",
        "us",
        "centers",
        "for",
        "disease",
        "control",
        "and",
        "prevention",
    ]


def test_keep_alphabetical_words():
    """Should return only alphabetical words."""
    answer = keep_alphabetical_words(
        sent=["the", "us", "centers", "for", "10", "302029", "disease"]
    )

    assert answer == [
        "the",
        "us",
        "centers",
        "for",
        "disease",
    ]


def test_tokenize_file(tmp_path):
    """Should tokenize a list of sentences."""
    (tmp_path / "text_file.txt").write_text(
        "The U.S. Centers for Disease Control and Prevention initially advised school systems to close if outbreaks occurred , then reversed itself , saying the apparent mildness of the virus meant most schools and day care centers should stay open , even if they had confirmed cases of swine flu . \nWhen Ms. Winfrey invited Suzanne Somers to share her controversial views about bio-identical hormone treatment on her syndicated show in 2009 , it won Ms. Winfrey a rare dollop of unflattering press , including a Newsweek cover story titled ' Crazy Talk : Oprah , Wacky Cures & You . "
    )

    answer = tokenize_file(file=str(tmp_path / "text_file.txt"))

    assert answer == [
        [
            "the",
            "us",
            "centers",
            "for",
            "disease",
            "control",
            "and",
            "prevention",
            "initially",
            "advised",
            "school",
            "systems",
            "to",
            "close",
            "if",
            "outbreaks",
            "occurred",
            "then",
            "reversed",
            "itself",
            "saying",
            "the",
            "apparent",
            "mildness",
            "of",
            "the",
            "virus",
            "meant",
            "most",
            "schools",
            "and",
            "day",
            "care",
            "centers",
            "should",
            "stay",
            "open",
            "even",
            "if",
            "they",
            "had",
            "confirmed",
            "cases",
            "of",
            "swine",
            "flu",
        ],
        [
            "when",
            "ms",
            "winfrey",
            "invited",
            "suzanne",
            "somers",
            "to",
            "share",
            "her",
            "controversial",
            "views",
            "about",
            "bioidentical",
            "hormone",
            "treatment",
            "on",
            "her",
            "syndicated",
            "show",
            "in",
            "it",
            "won",
            "ms",
            "winfrey",
            "a",
            "rare",
            "dollop",
            "of",
            "unflattering",
            "press",
            "including",
            "a",
            "newsweek",
            "cover",
            "story",
            "titled",
            "crazy",
            "talk",
            "oprah",
            "wacky",
            "cures",
            "you",
        ],
    ]


def test_flatten_list():
    """Should return a list made of a list of lists."""
    answer = flatten_list(list_of_lists=[["A", "B", "C"], ["D"], ["E", "F", "G"]])

    assert answer == ["A", "B", "C", "D", "E", "F", "G"]


def test_count_words():
    """Return a Counter of words in list."""
    answer = count_words(["A", "A", "B", "B", "A", "A", "C"])

    assert answer == Counter({"A": 4, "B": 2, "C": 1})
