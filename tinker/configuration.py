"""Utilities for managing Tinker meta-configuration files."""

import itertools
from typing import Dict, Any, Iterator, List
from typing_extensions import TypedDict
import yaml

from pprint import pprint


Replacer = TypedDict("Replacer", {
    "prefix": str,
    "iterator": Iterator[Any],
})


def set_dict_path(d: Dict, path: List[str], value: Any) -> Dict:
    """Set a "deep" value in a dict."""
    v = d
    for p in path[:-1]:
        v = v[p]
    v[path[-1]] = value


def config_walk(prefix: List[str], config: Dict) -> List[Iterator[Any]]:
    replacers = []
    skeleton = {}

    for key, value in config.items():
        new_prefix = prefix + [key]

        if type(value) == dict:
            if "iterate" in value:
                set_dict_path(skeleton, new_prefix, None)
                replacers.append((x for x in zip(value["iterate"], itertools.repeat(new_prefix))))
            else:
                replacers += config_walk(new_prefix, value)
        else:
            set_dict_path(skeleton, new_prefix, value)


    return {
        "replacers": replacers,
        "skeleton": skeleton,
    }


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
