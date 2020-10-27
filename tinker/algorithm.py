"""
Provide an interface for all algorithm implementations.

from abc import abstractmethod
import smqtk


class Algorithm(smqtk.algorithms.SmqtkAlgorithm):
    @classmethod
    def is_usable(cls):
        return True

    @abstractmethod
    def execute(self):
        pass
