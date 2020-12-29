"""Utilities for managing Tinker meta-configuration files."""

import itertools
from typing import Dict, Any, Iterator, List
from typing_extensions import TypedDict
import yaml


def is_iterate(value):
    """Return `True` if `value` is an "iterate" directive."""
    return type(value) is dict and "iterate" in value


def dict_permutations(d):
    """
    Generate all combinatorial permutations of a dict.

    The input is a dict `d` mapping its keys to generators containing all the
    iterations for that key. This function runs the product of all the
    generators in order to instantiate each version of the dict.
    """
    iterators = (zip(itertools.repeat(k), config_generator(v)) for k, v in d.items())
    for combo in itertools.product(*iterators):
        t = {}
        for k, v in combo:
            t[k] = v
        yield t


def singleton(v):
    """
    Create a singleton generator for a value.

    While this seems odd, its purpose is to allow *all* configuration values to
    be expressed as generators, simplifying the implementation of
    `dict_permutations()` above.
    """

    return (x for x in [v])


def iterate_generator(iterates):
    """
    Generate all values specificed by an "iterator" directive.

    The idea is to collect all values represented by each item and linearize
    them into a single list of values. Any nested "iterate" directives, for
    instance, will unfold into a portion of this list.
    """
    return itertools.chain(*(config_generator(i) for i in iterates))


def config_generator(value):
    """
    Generate values for any type that may appear in the configuration object.

    This is the top-level "entry point" for meta-configuration expansion. It
    dispatches on the type passed to it, invoking the proper processing routine
    to generate partial results. The function is ultimately recursive, as those
    processing routines (listed above) examine the contents of the values passed
    to them, invoking this function again as needed.
    """
    if is_iterate(value):
        return iterate_generator(value["iterate"])
    elif type(value) is dict:
        return dict_permutations(value)
    else:
        return singleton(value)


def parse_configuration(text: str) -> Dict:
    """Read in and process the contents of a configuration file."""
    config = yaml.safe_load(text)
    return config_generator(config)
