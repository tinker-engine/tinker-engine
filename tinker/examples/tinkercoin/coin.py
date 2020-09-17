"""Coin implementations."""

import random


class Coin(object):
    """Base class for flippable coins."""

    random: random.Random

    def __init__(self, seed: int = 0) -> None:
        """Initialize with a seeded RNG."""
        self.random = random.Random(seed)

    def flip(self) -> bool:
        """Flip the coin: `true` means heads; `false` means tails."""
        raise RuntimeError("not implemented")


class FairCoin(Coin):
    """A fair coin."""

    def flip(self) -> bool:
        """Flip the coin: should come up heads as often as tails."""
        return self.random.random() < 0.5


class UnfairCoin(Coin):
    """A coin biased towards heads."""

    def flip(self) -> bool:
        """Flip the coin: should come up heads 3/4 of the time."""
        return self.random.random() < 0.75


class WeirdCoin(Coin):
    """A strange coin that changes its bias whenever it comes up tails."""

    threshold: float

    def __init__(self, seed: int = 0) -> None:
        """Initialize with a variable threshold"""
        super().__init__(seed)
        self.threshold = 0.25

    def flip(self) -> bool:
        """Flip the coin."""
        result = self.random.random() < self.threshold

        if not result:
            self.threshold = 1.0 - self.threshold

        return result
