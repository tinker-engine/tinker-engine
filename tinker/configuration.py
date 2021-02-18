"""Utilities for managing Tinker meta-configuration files."""

import itertools
from typing import Dict, Union, List, Any, Iterator, cast
from typing_extensions import TypedDict
import yaml

from smqtk_core import Pluggable

class IterateDirective(TypedDict):
    """Type of iterate directive."""

    # This `Any` should also be `ConfigEntry`; see comment below.
    iterate: List[Any]


# The two instances of `Any` should rightfully be `ConfigEntry`, but
# unfortunately, mypy does not currently support recursive structural typing:
# https://github.com/python/mypy/issues/731.
ConfigEntry = Union[IterateDirective, Dict[str, Any], List[Any], str, int, float]
Config = Dict[str, Any]


def is_iterate(value: Any) -> bool:
    """Return `True` if `value` is an "iterate" directive."""
    return type(value) is dict and "iterate" in value

def dict_permutations(d: Dict[str, ConfigEntry]) -> Iterator[Config]:
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


def singleton(v: ConfigEntry) -> Iterator[ConfigEntry]:
    """
    Create a singleton generator for a value.

    While this seems odd, its purpose is to allow *all* configuration values to
    be expressed as generators, simplifying the implementation of
    `dict_permutations()` above.
    """

    return (x for x in [v])


def iterate_generator(iterates: List[ConfigEntry]) -> Iterator[ConfigEntry]:
    """
    Generate all values specificed by an "iterator" directive.

    The idea is to collect all values represented by each item and linearize
    them into a single list of values. Any nested "iterate" directives, for
    instance, will unfold into a portion of this list.
    """
    return itertools.chain(*(config_generator(i) for i in iterates))

def is_smqtk(value: Any) -> bool:
    """Return `True` if `value` is a "smqtk" directive."""
    return type(value) is dict and "smqtk" in value

def smqtk_generator(smqtk_def):
    """
    Find implementation of and instantiate a SMQTK class with a config
    all values specified by "smqtk" directive.


    """
    
    # get available implementations 
    # (from environment variable and entrypoint extension)
    candidate_types = Pluggable.get_impls()
    
    # check if our class matches any of the implementations
    class_name = smqtk_def['class']
    matched_class = None
    for candidate in candidate_types:
        if candidate.__name__ == class_name:
            matched_class = candidate
            break
    if matched_class is None:
        raise ValueError('No SMQTK definition found: {}'.format(class_name))
    
    # Check with is_usable (inherits from Pluggable)
    if not matched_class.is_usable():
       raise ValueError('SMQTK impl {} not usable'.format(class_name))
 
    # TODO: add explicit checks that inherits from Pluggable and Configurable
    # rather than these implicit checks
   
    # TODO: maybe check to call is_valid_plugin

    smqtk_config = smqtk_def.get('config',{})

    # needs to be as list so it is iterable
    return [matched_class.from_config(smqtk_config)]

def config_generator(value: ConfigEntry) -> Iterator[ConfigEntry]:
    """
    Generate values for any type that may appear in the configuration object.

    This is the top-level "entry point" for meta-configuration expansion. It
    dispatches on the type passed to it, invoking the proper processing routine
    to generate partial results. The function is ultimately recursive, as those
    processing routines (listed above) examine the contents of the values passed
    to them, invoking this function again as needed.
    """
    if is_iterate(value):
        return iterate_generator(cast(IterateDirective, value)["iterate"])
    elif is_smqtk(value):
        return smqtk_generator(value['smqtk']) 
    elif type(value) is dict:
        return dict_permutations(cast(Dict[str, Any], value))
    else:
        return singleton(value)


def parse_configuration(text: str) -> Iterator[Config]:
    """Read in and process the contents of a configuration file."""
    config = yaml.safe_load(text)
    # This assert is for the typechecker.
    #
    # TODO: add schema validation for `config` so that an error occurs before
    # this assert if the type of `config` is incorrect.
    assert isinstance(config, dict)

    return dict_permutations(config)
