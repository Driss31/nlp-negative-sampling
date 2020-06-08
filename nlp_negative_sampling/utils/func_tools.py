"""Tools to process functions."""
from functools import reduce


def compose(*functions):
    """Combine n functions such that the result of each function is passed as the argument of the new function."""
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)
