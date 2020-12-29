"""Utilities for managing Tinker meta-configuration files."""

import itertools
from typing import Dict, Any, Iterator, List
from typing_extensions import TypedDict
import yaml


def is_iterate(value):
    return type(value) is dict and "iterate" in value


def dict_permutations(d):
    iterators = (zip(itertools.repeat(k), v) for k, v in d.items())
    for combo in itertools.product(*iterators):
        t = {}
        for k, v in combo:
            t[k] = v
        yield t


def config_generator(value):
    if is_iterate(value):
        return itertools.chain(*(config_generator(v) for v in value["iterate"]))
    elif type(value) is dict:
        t = {}
        for k, v in value.items():
            t[k] = config_generator(v)
        return dict_permutations(t)
    else:
        return (x for x in [value])


def parse_configuration(text: str) -> Dict:
    config = yaml.safe_load(text)

    return config_generator(config)
