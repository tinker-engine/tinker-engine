"""Definition of Tinker protocol."""

from abc import abstractmethod
import smqtk  # type: ignore


class Protocol(smqtk.algorithms.SmqtkAlgorithm):
    """The Tinker protocol class, defined as a SMQTK algorithm with its own interface."""

    # Make these protocol objects usable by default.
    @classmethod
    def is_usable(cls) -> bool:
        """Determine if this class will be detected by SMQTK's plugin utilities."""
        return True

    @abstractmethod
    def run_protocol(self) -> None:
        """Run the protocol."""
        pass
