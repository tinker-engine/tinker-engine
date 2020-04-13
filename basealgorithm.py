
import abc
"""
.. _basealgorithm.py:

basealgorithm.py
============

The BaseAlgorithm class is a base class that provides an interface for all algorithm implementations
Each implementation must implement the execute() function. The execute function acts a ageneric dispatcher
for calling subfunctions of the algorithm. This is done to simplify the auto-generation of template
algorithm files for researchers to use.
"""


class BaseAlgorithm(metaclass=abc.ABCMeta):
    """
    Base Class for defining an algorithm. Implementations of
    an alogirthm should inherit from this class to be run within
    the framework. Each implementation must implent the execute
    function.

    An algorithm object is persistent for the duration of the
    test protocol, so meta-information can be assigned to class
    members (self.xxxx), and will remain avialable for future
    use.

    Attributes:
        problem (LwLL): same as input
        toolset: This is a dictionary of named functions that
        provide basic services such as "GetDatasetList" and
        "GetPretrainedModelList"

    """
    def __init__(self, problem, arguments):
        self.problem = problem
        self.arguments = arguments

    @abc.abstractmethod
    def execute(self, toolset, step_descriptor):
        raise NotImplementedError


