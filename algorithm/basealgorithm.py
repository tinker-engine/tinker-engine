
import abc
"""
.. _algorithm.py:

basealgorithm.py
============

Here is where your code is added to the algorithm.
This code uses the VAAL algorithm as an example.
The rest of the files in this folder pertain to that
algorithm.

"""


class BaseAlgorithm(metaclass=abc.ABCMeta):
    """
    Class which runs the algorithm consisting of at least four
    methods: :meth:`__init__`, :meth:`train`, :meth:`adapt`,
    and :meth:`inference`.  You may have more classes if you
    want, but these are the ones called from the :ref:`main.py`
    file.

    This class requires an init, train, adapt, and
    inference methods.  These methods run at the different
    stages of the problem and are called from :ref:`main.py`.
    You can add more attributes or functions as you see fit.

    For example, you want to preserve the train stage task
    model during the adapt stage so you can add in
    ``self.train_task_model`` as an attribute.
    Alternatively, you can just
    save it in a helper class like "solver" here.

    Attributes:
        problem (LwLL): same as input
        base_dataset (JPLDataset): same as input
        current_dataset (JPLDataset): dataset currently in use
            (either the train or adapt stage dataset)
        adapt_dataset (JPLDataset): the adapt stage dataset
            (defined as None until that stage)
        arguments (dict[str, str]): same as input
        cuda (bool): whether or not to use cuda during inference
        vaal (Solver): solver class for VAAL code
        train_accuracies (list): training accuracies
        task_model (torch.nn.Module): pytorch model for bridging
            between train/adapt stages and the eval stage.

    The last four variables (after arguments) are specific to the VAAL algorithm
    so are not necessary for the overall problem.
    """
    def __init__(self):
        pass 

    @abc.abstractmethod
    def execute(self, stage):
        raise NotImplementedError

    @abc.abstractmethod
    def test( self, eval_dataset ):
        raise NotImplementedError

    #TODO: findDatasets funciton

    #TODO: getDataset function(s)

    #TODO: getBudget function

    #TODO: getLabels function


