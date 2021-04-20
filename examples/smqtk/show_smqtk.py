"""Meta-configuration demonstration."""

from tinker import protocol
from smqtk_core import Configurable


class ShowSmqtk(protocol.Protocol):
    """
    A protocol designed for testing SMQTK integration.

    Will initialize and then print out config / instance details
    """

    def get_config(self):
        """Return a default configuration dictionary."""
        return {}

    def run_protocol(self, config):
        """Run the protocol by printout out the config."""
        # TODO: make this recursive to support multiple-inheritance in the config
        for k, v in config.items():
            if isinstance(v, Configurable):
                v = v.__str__() + "\n\t" + (str(v.get_config()))
            print(k, " : ", v)
