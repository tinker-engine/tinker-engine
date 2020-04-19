
import abc
"""
.. _basealgorithm.py:

basealgorithm.py
============

The BaseAlgorithm class is a base class that provides an interface for all algorithm implementations
Each implementation must implement the execute() function. The execute function acts a generic dispatcher
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
        toolset: This is a dictionary of named functions that
        provide basic services such as "GetDatasetList" and
        "GetPretrainedModelList"

    """
    def __init__(self, toolset):
        if isinstance(toolset, dict):
            self.toolset= toolset
        elif toolset:
            print( "Algorithms must be constructed with dictionary toolset" )
            exit(1)
        else:
            toolset = dict()

    @abc.abstractmethod
    def execute(self, toolset, step_descriptor):
        '''
        The execute function is a generic dispatch function for handling calls to the algorithm
        This generic dispatch call allow the introduction of an adapter class for dispatching
        calls to pre-existing algorithms. This minimizes the changes that need to be made to
        an existing algorithm to adapt it for use in the framework.

        Arguments:
            toolset: This is a dict of named resources available for this step of execution.
            step_descriptor: This is a sting describing the desired step of execution to perform
                This is usually, but not always, the name of the function to call in the derived
                algorithm class.
        '''
        raise NotImplementedError


