"""
This file contains the main code to run the experiment.

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
import os
import pkg_resources
from pkg_resources import EntryPoint
import logging
import time
import socket

from typing import Any, Optional


def _safe_load(entry_point: EntryPoint) -> Any:
    """Load algorithms from an entrypoint without raising exceptions."""
    try:
        return entry_point.load()
    except Exception as fault:
        logging.error("Cannot load entrypoint")
        logging.exception(fault)
        exit(1)


discovered_plugins = {
    entry_point.name: _safe_load(entry_point) for entry_point in pkg_resources.iter_entry_points("tinker")
}


def main() -> None:
    """Run the main program."""

    # Setup the argument parsing, and generate help information.
    parser = argparse.ArgumentParser()
    parser.add_argument("entrypoints", metavar="entrypoint", nargs="+", help="python file defining protocols/algorithms/etc.", type=str)
    parser.add_argument("-c", "--config", help="config file", type=str, required=True)
    parser.add_argument(
        "--list-protocols", help="Print the available protocols"
    )
    parser.add_argument(
        "--list-algorithms", help="Print the available algorithms"
    )
    parser.add_argument(
        "--log-file",
        help="Path to log file",
        type=str,
        default=f"tinker_{socket.gethostname()}_{time.asctime().replace(' ', '_')}.log",
    )
    parser.add_argument("--log-level", default=logging.INFO, help="Logging level", type=int)

    args = parser.parse_args()

    # Set up the logger.
    log_format = "[tinker-engine] %(asctime)s %(message)s"
    logging.basicConfig(filename=args.log_file, filemode="w", level=args.log_level, format=log_format)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # Find the config file.
    if not os.path.exists(args.config):
        logging.error(f"error: config file {args.config} doesn't exist")
        return 1

    # list the available interfaces
    harness = None
    if args.list_interfaces:
        # print the interfaces included with Tinker Engine.
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            print_interface(name, obj)

        # print the interfaces in the protocol directory
        check_directory_for_interface(protocol_file_path, args.interface, True)

        # print the interfaces in the current working directory.
        check_directory_for_interface(".", args.interface, True)

        # print any interfaces available in the plugins
        for name in discovered_plugins.keys():
            print_interface(name, discovered_plugins[name])

        # after printing the interfaces, there is nothing else to do.
        exit(0)

    # search for the desired interface in the various places it could be.
    if harness is None:
        # check the protocol directory for the desired interface class
        harness = check_directory_for_interface(protocol_file_path, args.interface, False)
    if harness is None:
        # check the current working directory for the desired interface class.
        harness = check_directory_for_interface(".", args.interface, False)

    # check the plugins for a Harness that matches the interface argument
    if harness is None:
        obj = discovered_plugins.get(args.interface)
        if obj is not None:
            harness = obj("configuration.json", protocol_file_path)

    # as a last resort, look for the interface in Tinker Engine itself.
    if harness is None:
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if args.interface == name and inspect.isclass(obj) and issubclass(obj, Harness):
                harness = obj("configuration.json", protocol_file_path)

    if harness is None:
        logging.error("Requested interface not found")
        exit(1)

    protbase, protext = os.path.splitext(protfile)

    # make sure the protocol file is a python file
    if protext == ".py":
        # import the file and get the object name. The object should go in the
        # protocol local object
        protocolimport = __import__(protbase, globals(), locals(), [], 0)
        for _name, obj in inspect.getmembers(protocolimport):
            # This will get every class that is referenced within the file,
            # including base classes to ensure we get the right one, check for only
            # classes that are within the module defined by the protocol file.
            if inspect.isclass(obj):
                foo = inspect.getmodule(obj)
                if foo == protocolimport:
                    # construct the protocol object
                    protocol = obj(discovered_plugins, algorithmsbasepath, harness, config_file)
    else:
        logging.error("Invalid protocol file, must be a python3 source file")
        exit(1)

    if protocol:
        protocol.run_protocol()
    else:
        logging.error("protocol invalid")
        exit(1)


def check_directory_for_interface(file_path: str, interface_name: str, print_interfaces: bool) -> Optional[Any]:
    """
    Load Harness objects found on the Python path.

    Check the given file path for any python files containing classes that derive
    from the Harness class. If the print_interfaces flag is set, then print any that
    are found. If that flag is not set, then instantiate the interface name interface_name,
    and return the object.
    """
    harness = None
    for file in os.listdir(file_path):
        try:
            filebase, fileext = os.path.splitext(file)
            if fileext == ".py" and not filebase == "__init__" and not filebase == "setup":
                interfaceimport = __import__(filebase, globals(), locals(), [], 0)
                for name, obj in inspect.getmembers(interfaceimport):
                    if inspect.isclass(obj) and interfaceimport == inspect.getmodule(obj):
                        if print_interfaces:
                            print_interface(name, obj)
                        elif name == interface_name and issubclass(obj, Harness):
                            harness = obj("configuration.json", file_path)
        except Exception:
            # ignore any import error, but leave the harness set to none to indicate failure
            continue
    return harness


def print_interface(name: str, obj: Any) -> None:
    """Print out information about an interface."""
    if inspect.isclass(obj):
        if issubclass(obj, Harness) and not name == "Harness":
            print(name, obj)


if __name__ == "__main__":
    main()
