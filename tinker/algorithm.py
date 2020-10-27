"""Definition of Tinker algorithm."""

from abc import abstractmethod
import smqtk


class Algorithm(smqtk.algorithms.SmqtkAlgorithm):
    """The Tinker algorithm class, defined as a SMQTK algorithm with its own interface."""

    @classmethod
    def is_usable(cls):
        """Determine if this class will be detected by SMQTK's plugin utilities."""
        return True

    @abstractmethod
    def execute(self):
        """Run the algorithm."""
        pass
