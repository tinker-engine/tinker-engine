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
from framework.harness import Harness
from framework.localinterface import LocalInterface
from framework.jplinterface import JPLInterface

protocol_file_path = ""

def execute(req):
    # Setup the argument parsing, and generate help information.
    global protocol_file_path
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
    parser.add_argument("-i", "--interface",
            help="Name of the Interface class to use. use '--list_interfaces' to show available interfaces",
            type= str,
            default = "LocalInterface")
    parser.add_argument("-l", "--list_interfaces",
            help="Print the list of available interfaces",
            action='store_true')

    args = parser.parse_args()

    # TODO: implement the --generate functionality

    # Check the algorithms path is minimally acceptable.
    algorithmsbasepath = args.algorithms
    if not os.path.exists(algorithmsbasepath):
        print(f"algorithm directory {algorithmsbasepath} doesn't exist")
        exit(1)

    if not os.path.isdir(algorithmsbasepath):
        print(f"algorithm path {algorithmsbasepath} isn't a directory")
        exit(1)

    # deconstruct the path to the protocol so that we can construct the
    # object dynamically.
    protfilename = args.protocol_file
    if not os.path.exists(protfilename):
        print(f"protocol file {protfilename} does not exist")
        sys.exit(1)

    # split out the path to the protocol file from the filename so that we can add
    # the protocol directory
    # to the system path.
    protocol_file_path, protfile = os.path.split(protfilename);

    if protocol_file_path:
        sys.path.append(protocol_file_path)



    # find the interface (or list the available ones)
    harness = None
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if args.list_interfaces:
            print_interface( name, obj)
        else:
            if args.interface == name and inspect.isclass(obj) and issubclass(obj, Harness):
                harness = obj('configuration.json', protocol_file_path)
    #TODO: check the working direcotory and "protocol" directory as well.

    if args.list_interfaces:
        # nothing more to do here, all we are doing is listing the itnerfaces.
        exit(0)

    if harness is None:
        print( "Interface not found" )
        exit(1)


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
                    protocol = obj(algorithmsbasepath, harness)
    else:
        print("Invalid protocol file, must be a python3 source file")
        sys.exit(1)

    if protocol:
        protocol.run_protocol()
    else:
        print("protocol invalid")

def print_interface( name, obj):
    if inspect.isclass(obj):
       if issubclass(obj, Harness) and not name == "Harness":
           print(name, obj)

    

def main():
    """ Main to run the algorithm locally.  Just loads the input.json file and calls
    the :meth:`main.execute` function.
    """
    execute({})


if __name__ == "__main__":
    main()
