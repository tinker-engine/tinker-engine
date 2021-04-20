"""Simple SMQTK demonstration."""

from smqtk_core import Pluggable, Configurable
import json


class HelloWorldSmqtk(Pluggable, Configurable, json.JSONEncoder):
    r"""
    A simple SMQTK implementation of Pluggable and Configurable.

    For this to be found, it needs to be discoverable.
    The easiest way to do this is via environmental variables.
    For example, from the root directory, set SMQTK_PLUGIN_PATH=examples.smqtk.hello_smqtk

    Example of how to run:
    SMQTK_PLUGIN_PATH=examples.smqtk.hello_smqtk tinker \
        -c examples/smqtk/hello_smqtk.yaml examples/smqtk/show_smqtk.py
    """

    def __init__(self, foo=1, bar=2, baz=3):
        """Initialize by storing passed arguments."""
        self.foo = foo
        self.bar = bar
        self.baz = baz
        super().__init__()

    def get_config(self):
        """Fulfill required implementation from Configurable."""
        return {"foo": self.foo, "bar": self.bar, "baz": self.baz}
