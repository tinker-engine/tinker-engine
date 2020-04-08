
import abc
"""
.. _basealgorithm.py:

basealgorithm.py
============

The BaseAlgorithm class is a base class that provides an interface for all algorithm implementations
Each implementation must implement the execute() and train() functions. The execute function does not
return any information. It is meant for active learning, and other processing that evolves the internal
model.
The test() function is not intended to modify the model in any way, but that is not prohibited. The test
should return information that allows the objective evaluation of the perofrmance of the overall algorithm
(e.g. detection set or ).
"""


class BaseAlgorithm(metaclass=abc.ABCMeta):
    """
    Base Class for defining an algorithm. Implementations of
    an alogirthm should inherit from this class to be run within
    the framework. Each implementation must implent the execute
    and test functions.

    Class which runs the algorithm consisting of at least four
    methods: :meth:`__init__`, :meth:`train`, :meth:`adapt`,
    and :meth:`inference`.  You may have more classes if you
    want, but these are the ones called from the :ref:`main.py`
    file.

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
    def __init__(self, problem, toolset):
        self.problem = problem
        self.toolset = toolset

    @abc.abstractmethod
    def execute(self, step_descriptor):
        raise NotImplementedError

    @abc.abstractmethod
    def test( self, eval_dataset, step_descriptor ):
        raise NotImplementedError


