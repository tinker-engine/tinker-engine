"""A Tinker Engine protocol for testing flippable coins."""

from CoinFlipper import WeightedCoin
import tinker.protocol


class CoinFlip(tinker.protocol.Protocol):
    """
    A coin flipping protocol.

    It works by constructing a weighted coin of a certain weighting, then
    flipping it a specific number of times, reporting both the number of heads
    that came up and the statistically expected number of heads.
    """

    def get_config(self):
        """Return a default configuration."""
        return {}

    def run_protocol(self, config) -> None:
        """Run the coin flipping experimental protocol."""

        # Extract the config values.
        weight = config["weight"]
        trials = config["trials"]

        # Construct a coin flipper with the specified weight.
        flipper = WeightedCoin(config["weight"])

        # Run the trials.
        heads = sum(1 if flipper.flip() else 0 for _ in range(trials))

        # Display the result.
        print(f"{heads} heads out of {trials} flips; expected {trials * weight}")
