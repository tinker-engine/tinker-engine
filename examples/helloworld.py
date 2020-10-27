"""Protocol that implements hello world functionality."""
from tinker import protocol


class MyProtocol(protocol.Protocol):
    """Protocol that implements hello world functionality."""

    def get_config(self):
        """Config required by the ``smqtk.utils.configuration.Configurable`` parent."""

        return {}

    def run(self):
        """Run state of the protocol."""

        print("hello, world")
