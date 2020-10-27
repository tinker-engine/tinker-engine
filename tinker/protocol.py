"""Base protocol class."""

from abc import abstractmethod
import smqtk  # type: ignore


class Protocol(smqtk.algorithms.SmqtkAlgorithm):
    """Base protocol class."""

    @classmethod
    def is_usable(cls) -> bool:
        """Usable method from ``smqtk.utils.configuration.Configurable`` parent."""

        return True

    @abstractmethod
    def run(self) -> None:
        """Runtime for the protocol."""

        pass
