"""Definition of Tinker algorithm."""

from abc import abstractmethod
import smqtk  # type: ignore


class Algorithm(smqtk.algorithms.SmqtkAlgorithm):
    """The Tinker algorithm class, defined as a SMQTK algorithm with its own interface."""

    @classmethod
    def is_usable(cls) -> bool:
        """Determine if this class will be detected by SMQTK's plugin utilities."""
        return True

    @abstractmethod
    def execute(self) -> None:
        """Run the algorithm."""
        pass
