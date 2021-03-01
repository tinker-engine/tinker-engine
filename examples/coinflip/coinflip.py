from CoinFlipper import WeightedCoin
import tinker.protocol


class CoinFlip(tinker.protocol.Protocol):
    def get_config(self):
        return {}

    def run_protocol(self, config) -> None:
        # Extract the config values.
        weight = config["weight"]
        trials = config["trials"]

        # Construct a coin flipper with the specified weight.
        flipper = WeightedCoin(config["weight"])

        # Run the trials.
        heads = sum(1 if flipper.flip() else 0 for _ in range(trials))

        # Display the result.
        print(f"{heads} heads out of {trials} flips; expected {trials * weight}")
