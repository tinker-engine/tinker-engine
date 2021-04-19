"""Utilities for managing Tinker meta-configuration files."""

import itertools
from typing import Dict, Union, List, Any, Iterator, cast
from typing_extensions import TypedDict
import yaml

from smqtk_core import Pluggable, Configurable


class IterateDirective(TypedDict):
    """Type of iterate directive."""

    # This `Any` should also be `ConfigEntry`; see comment below.
    iterate: List[Any]


class SMQTKDirective(TypedDict):
    """Type of smqtk directive."""

    # This `Any` should also be `ConfigEntry`; see comment below.
    smqtk: Dict[str, Any]


# The two instances of `Any` should rightfully be `ConfigEntry`, but
# unfortunately, mypy does not currently support recursive structural typing:
# https://github.com/python/mypy/issues/731.
# Also, the instance of Pluggable or Configurable probably should instead be
# Pluggable AND Configurable
ConfigEntry = Union[
    SMQTKDirective, IterateDirective, Dict[str, Any], List[Any], str, int, float, Pluggable, Configurable
]
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
    iterators = (zip(itertools.repeat(k), preprocess_config_generator(v)) for k, v in d.items())
    for combo in itertools.product(*iterators):
        t = {}
        for k, v in combo:
            t[k] = process_config(v)
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
    return itertools.chain(*(preprocess_config_generator(i) for i in iterates))


def is_smqtk(value: Any) -> bool:
    """Return `True` if `value` is a "smqtk" directive."""
    return type(value) is dict and "smqtk" in value


def smqtk_generator(smqtk_def: Dict[str, ConfigEntry]) -> ConfigEntry:
    r"""
    Find implementation of and instantiate a SMQTK class.

    Instantiated with a config all values specified
    by "smqtk" directive.
    """

    # get available implementations
    # (from environment variable and entrypoint extension)
    candidate_types = Pluggable.get_impls()

    # check if our class matches any of the implementations
    class_name = smqtk_def["class"]
    matched_class = None  # TODO: can add typing here for matched_class
    for candidate in candidate_types:
        if candidate.__name__ == class_name:
            matched_class = candidate
            # TODO: add some sort of error checking if multiple classes
            # have the same name, so we don't have any unexpected
            # behavior
            break
    if matched_class is None:
        raise ValueError("No SMQTK definition found: {}".format(class_name))

    # explict check that matched_class is Configurable
    if not issubclass(matched_class, Configurable):
        raise ValueError("{} must be Configurable".format(class_name))

    # Check with is_usable (inherits from Pluggable)
    if not matched_class.is_usable():
        raise ValueError("SMQTK impl {} not usable".format(class_name))

    # TODO: maybe check to call is_valid_plugin

    smqtk_config = smqtk_def.get("config", {})

    smqtk_impl = matched_class.from_config(cast(Dict[str, Any], smqtk_config))
    return smqtk_impl


def process_config(value: ConfigEntry) -> ConfigEntry:
    """
    Process any directives not specified in 'preprocess_config_generator'.

    After 'preprocess_config_generator' is called and expands the config,
    this will process any other directive.
    """
    if is_smqtk(value):
        return smqtk_generator(cast(SMQTKDirective, value)["smqtk"])
    else:
        return value


def preprocess_config_generator(value: ConfigEntry) -> Iterator[ConfigEntry]:
    """
    Generate values for any type that may appear in the configuration object.

    This is the top-level "entry point" for meta-configuration expansion. It
    dispatches on the type passed to it, invoking the proper preprocessing routine
    to generate partial results. The function is ultimately recursive, as those
    processing routines (listed above) examine the contents of the values passed
    to them, invoking this function again as needed.
    """
    if is_iterate(value):
        return iterate_generator(cast(IterateDirective, value)["iterate"])
    elif type(value) is dict:
        return dict_permutations(cast(Dict[str, Any], value))
    # TODO: add support for directive in list / tuple -
    # some sort of "list_permutations" function
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
