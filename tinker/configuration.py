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


def parse_configuration(text: str) -> Dict:
    config = yaml.safe_load(text)

    # Perform a recursive descent down the values in the config object, looking
    # for signal words.
    plan = config_walk([], config)

    all_combos = itertools.product(*plan["replacers"])
    for combo in all_combos:
        config = dict(plan["skeleton"])
        for replacement in combo:
            set_dict_path(config, replacement[1], replacement[0])

        yield(config)
