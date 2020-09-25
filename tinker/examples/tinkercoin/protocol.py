"""Protocol for testing of coin fairness."""

from pprint import pprint
from tinker.baseprotocol import BaseProtocol

from coin import Coin, FairCoin, UnfairCoin, WeirdCoin


class FairCoinTest(BaseProtocol):
    """Tinker Engine protocol for testing coin fairness."""

    def test_coin(self, name: str, coin: Coin, num_trials: int):
        heads = 0
        tails = 0
        for _ in range(num_trials):
            if coin.flip():
                heads += 1
            else:
                tails += 1

        print(f"{name}: {heads} heads, {tails} tails")

    def run_protocol(self):
        self.test_coin("FairCoin", FairCoin(), 10000)
        self.test_coin("UnfairCoin", UnfairCoin(), 10000)
        self.test_coin("WeirdCoin", WeirdCoin(), 10000)
