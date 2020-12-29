"""Utilities for managing Tinker meta-configuration files."""

import itertools
from typing import Dict, Any, Iterator, List
from typing_extensions import TypedDict
import yaml


def is_iterate(value):
    return type(value) is dict and "iterate" in value


def dict_permutations(d):
    iterators = (zip(itertools.repeat(k), config_generator(v)) for k, v in d.items())
    for combo in itertools.product(*iterators):
        t = {}
        for k, v in combo:
            t[k] = v
        yield t


def singleton(v):
    return (x for x in [v])


def iterate_generator(iterates):
    return itertools.chain(*(config_generator(i) for i in iterates))


def config_generator(value):
    if is_iterate(value):
        return iterate_generator(value["iterate"])
    elif type(value) is dict:
        return dict_permutations(value)
    else:
        return singleton(value)


def parse_configuration(text: str) -> Dict:
    config = yaml.safe_load(text)

    return config_generator(config)
