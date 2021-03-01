"""A framework for simulating coin flips."""

from abc import ABC, abstractmethod
import random


class CoinFlipper(ABC):
    """An abstract base class for all flippable coins."""

    @abstractmethod
    def flip(self) -> bool:
        """Flip the coin. `True` means heads and `False` means tails."""
        pass


class WeightedCoin(CoinFlipper):
    """A weighted coin."""

    def __init__(self, weight: float = 0.5) -> None:
        """Initialize a weighted coin that comes up heads with probability `weight`."""
        self.weight = weight

    def flip(self) -> bool:
        """Flip the coin."""
        return random.random() < self.weight
