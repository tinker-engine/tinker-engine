"""Meta-configuration demonstration."""

from tinker import protocol


class MyProtocol(protocol.Protocol):
    """A protocol demonstrating how meta-configurations work."""

    def get_config(self):
        """Return a default configuration dictionary."""
        return {}

    def run_protocol(self, config):
        """Run the protocol by printout out the config."""
        print(config)
