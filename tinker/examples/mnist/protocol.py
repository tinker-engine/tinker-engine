"""Example Protocol."""
from tinker.baseprotocol import BaseProtocol

from mnist import train, evaluate


class Mnist(BaseProtocol):
    """Tinker Engine protocol to train and evaluate on MNIST."""

    def run_protocol(self):
        """Train and evaluate on MNIST."""

        # Train
        train(1, "mnist_model_example")

        # Eval 1
        evaluate(4444, "mnist_model_example")

        # Eval 2
        evaluate(500, "mnist_model_example")
