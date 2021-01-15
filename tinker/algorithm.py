"""Definition of Tinker algorithm."""

from abc import abstractmethod
from smqtk_core import Configurable, Pluggable


class Algorithm(Configurable, Pluggable):
    """The Tinker algorithm class, defined as a SMQTK algorithm with its own interface."""

    @classmethod
    def is_usable(cls) -> bool:
        """Determine if this class will be detected by SMQTK's plugin utilities."""
        return True

    @abstractmethod
    def execute(self) -> None:
        """Run the algorithm."""
        pass
