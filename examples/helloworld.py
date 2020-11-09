"""A minimal example of a Tinker protocol."""

from tinker import protocol


class MyProtocol(protocol.Protocol):
    """A very simple protocol that does next to nothing."""

    def get_config(self):
        """Return a default configuration dictionary."""
        return {}

    def run_protocol(self):
        """Run the protocol by printing a message."""
        print("hello, world")
