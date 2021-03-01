from abc import ABC, abstractmethod
import random


class CoinFlipper(ABC):
    @abstractmethod
    def flip(self) -> bool:
        pass


class WeightedCoin(CoinFlipper):
    def __init__(self, weight: float = 0.5) -> None:
        self.weight = weight

    def flip(self) -> bool:
        return random.random() < self.weight
