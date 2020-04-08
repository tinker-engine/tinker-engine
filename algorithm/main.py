"""
.. _main.py:

main.py
==========

This file contains the main code to run the experiment.  The execute function
runs the entire LEARN problem including:

    - Get the problem ID
        - First one picked at the moment
    - Initializes the problem as an :class:`problem.LwLL` class instance
        - This will create the session and download the dataset
    - Initializes the Base Datasets (both for training and evaluation)
        - This will automatically get the seed labels in the init
    - Initializes the algorithm
        - Creates :class:`algorithm.Algorithm` object
    - Runs Train Stage Loop
        - Calls method :meth:`algorithm.Algorithm.train`
    - Initializes Adapt Stage Datasets
        - Sets up adapt datasets
    - Runs Adapt Stage Loop
        - Same as train, but uses :meth:`algorithm.Algorithm.adapt`

This function runs the game and shouldn't be changed for any algorithm.
Email Kitware if you think that this needs to be changed.
(christopher.funk@kitware.com and eran.swears@kitware.com)

"""

import json
import os
import requests
from problem import LwLL
from example.simple import ExampleAlgorithm
import algorithm
import algorithm_objdet
from dataset import JPLDataset, JPLEvalDataset


def execute(req):
    """
    Args:
        req (dict): The input json as a dictionary in this dictionary
            called arguments

            Example: :

                >>> from main import *
                >>> req = {}
                >>> with open('input.json') as json_file:
                >>>     req['arguments'] = json.load(json_file)



    """

    # #### Get Secret/URL #####
    try:
        secret = os.environ['JPL_API_SECRET']
        url = os.environ['JPL_API_URL']
    except KeyError:
        raise RuntimeError('Need to have secrets file sourced ($source secrets)')

    # #### Get the list of problems from JPL and select the first one #####
    headers = {'user_secret': secret}
    r = requests.get(f"{url}/list_tasks", headers=headers)
    r.raise_for_status()
    dataset_dir = req['arguments']['dataset_dir']

    # problem_id = r.json()['tasks'][0]
    problem_id = 'problem_test_obj_detection'
    for problem_id in ['problem_test_image_classification',
                       'problem_test_obj_detection']:
        # ############## Initialize the Problem Class ###################
        #  This is a class which holds all the data about the problem
        #  Calling this will initialize the problem
        problem = LwLL(secret, url, problem_id, dataset_dir=dataset_dir)
        problem.download_dataset()  # Download the dataset
        print(problem)

        # ############# Initialize the Base Datasets ##################
        #  This code initials the train and eval datasets
        base_train_dataset = JPLDataset(problem)
        print(base_train_dataset)
        base_eval_dataset = JPLEvalDataset(problem, base_train_dataset)
        print(base_eval_dataset)

        traintoolset = {}
        evaltoolset = {}

        traintoolset["Dataset"] = base_train_dataset
        evaltoolset["Dataset"] = base_eval_dataset
        # ############# Initialize the Algorithm ##################
        #    You will edit the algorithm.py to initialize your algorithm
        #    however you see fit
        if problem.task_metadata['problem_type'] == 'image_classification':
            from example.image_classification import ImageClassifierAlgorithm
            alg = ImageClassifierAlgorithm(problem, req["arguments"])
        elif problem.task_metadata['problem_type'] == 'object_detection':
            from example.object_detector import ObjectDetectorAlgorithm
            alg = ObjectDetectorAlgorithm(problem, req["arguments"])

        # ############# Run Train Stage Loop ##################
        #  Run the training code for the number of budget levels
        #  Evaluate when done training with the labelsa
        print('Train ' + problem.format_status())
        num_budgets = len(problem.get_current_status['current_label_budget_stages'])
        for budget_idx in range(num_budgets):
            print(f'Train Stage {budget_idx}')
            # ############### Run the train function for a single budget ##########
            #   This should be run until you have used up all your labeling budget
            #       and are finished training with the current labels
            alg.execute(traintoolset, "train") 

            # ############### Evaluate and submit the labels ################
            #    Run forward inference on the labels and submit them to JPL
            preds, indices = alg.test(evaltoolset, "inference")
            base_eval_dataset.submit_predictions(preds, indices)

            print(problem.format_status() + '\n')

        # ############# Initialize the Adapt Datasets ##################
        # Initialize the adaption dataset now we are in the adaption state
        adapt_train_dataset = JPLDataset(problem,
                                         baseDataset=False)
        adapt_eval_dataset = JPLEvalDataset(problem,
                                            adapt_train_dataset,
                                            baseDataset=False)
        # ############# Update the Algorithm's Datasets ##################
        # Updating the current dataset and the adapt dataset so that the algorithm
        # can use it.
        traintoolset["Dataset"] = adapt_train_dataset
        evaltoolset["Dataset"] = adapt_eval_dataset

        # ############# Run Adapt Stage Loop ##################
        #  Run the adapt code for the number of budget levels
        #  Evaluate when done training with the labels
        print('Adapt ' + problem.format_task_metadata())
        problem.format_status()
        num_budgets = len(problem.get_current_status['current_label_budget_stages'])
        for budget_idx in range(num_budgets):
            print(f'Adapt Stage {budget_idx}')
            # ############### Run the adapt function for a single budget ##########
            #   This should be run until you have used up all your labeling budget
            #       and are finished training with the current labels
            alg.adapt(traintoolset, "adapt")

            # ############### Evaluate and submit the labels ################
            #    Run forward inference on the labels and submit them to JPL
            preds, indices = alg.test(,evaltoolset, "inference")
            adapt_eval_dataset.submit_predictions(preds, indices)

            print(problem.format_status(update=False) + '\n')

        print(problem.format_task_metadata())


def main():
    """ Main to run the algorithm locally.  Just loads the input.json file and calls
    the :meth:`main.execute` function.
    """
    with open('input.json') as json_file:
        execute({'arguments': json.load(json_file)})


if __name__ == "__main__":
    main()
