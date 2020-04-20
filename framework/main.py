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
import inspect
import sys
import argparse
import json
import os
import requests
from framework.dataset import JPLDataset


def execute(req):
    # Setup the argument parsing, and generate help information.
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol_file",
            help="protocol python file",
            type= str)
    parser.add_argument("-a", "--algorithms",
            help="root of the algorithms directory",
            type= str,
            default = ".")
    parser.add_argument("-g", "--generate",
            help="Generate template algorithm files",
            action='store_true')

    args = parser.parse_args()

    # TODO: implement the --generate functionality

    # Check the algorithms path is minimally acceptable.
    algorithmsbasepath = args.algorithms
    if not os.path.exists(algorithmsbasepath):
        print("given algorithm directory doesni't exist")
        exit(1)

    if not os.path.isdir(algorithmsbasepath):
        print("given algorithm path isnt a directory")
        exit(1)

    # deconstruct the path to the protocol so that we can construct the
    # object dynamically.
    protfilename = args.protocol_file
    if not os.path.exists(protfilename):
        print("given protocol file does not exist")
        sys.exit(1)

    # split out the path to the protocol file from the filename so that we can add
    # the protocol directory
    # to the system path.
    protpath, protfile = os.path.split(protfilename);
    if protpath:
        sys.path.append(protpath)
    protbase, protext = os.path.splitext(protfile)

    # make sure the protocol file is a python file
    if protext == ".py":
        # import the file and get the object name. The object should go in the
        # protocol local object
        protocolimport = __import__(protbase, globals(), locals(), [], 0)
        for name, obj in inspect.getmembers(protocolimport):
            # This will get every class that is referenced within the file,
            # including base classes to ensure we get the right one, check for only
            # classes that are within the module defined by the protocol file.
            if inspect.isclass(obj):
                foo = inspect.getmodule( obj )
                if foo == protocolimport:
                    #construct the protocol object
                    protocol = obj(algorithmsbasepath)
    else:
        print("Invalid protocol file, must be a python3 source file")
        sys.exit(1)

    if protocol:
        protocol.run_protocol()
    else:
        print("protocol invalid")


def main():
    """ Main to run the algorithm locally.  Just loads the input.json file and calls
    the :meth:`main.execute` function.
    """
    execute({})


if __name__ == "__main__":
    main()
