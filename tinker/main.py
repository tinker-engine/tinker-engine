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
import argparse
import importlib.util
import logging
import os
import pkg_resources
from pkg_resources import EntryPoint
import smqtk
import socket
import sys
import time
from typing import Any, List

from . import algorithm
from . import protocol


def import_source(path: str) -> None:
    """
    Import a module, identified by its path on disk.

    Arguments:
        path: Absolute or relative path to the file of the Python module to import.
    """

    # Extract the name portion of the path.
    basename = os.path.basename(path)
    module_name = os.path.splitext(basename)[0]

    # Run the Python 3 recipe for programmatic importing of a source path (see
    # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly).
    spec = importlib.util.spec_from_file_location(module_name, path)

    # This is a typechecking workaround; see
    # https://github.com/python/typeshed/issues/2793.
    assert isinstance(spec.loader, importlib.abc.Loader)

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


def _safe_load(entry_point: EntryPoint) -> Any:
    """Load algorithms from an entrypoint without raising exceptions."""
    try:
        return entry_point.load()
    except Exception as fault:
        logging.error("Cannot load entrypoint")
        logging.exception(fault)
        exit(1)


def print_objects(objects: List[smqtk.algorithms.SmqtkAlgorithm], title: str) -> None:
    print(f"{title}:")
    if not objects:
        print("  none found")
    else:
        for o in objects:
            print(f"  {o.__module__}.{o.__name__}")


discovered_plugins = {
    entry_point.name: _safe_load(entry_point) for entry_point in pkg_resources.iter_entry_points("tinker")
}


def main() -> int:
    """Run the main program."""

    # Setup the argument parsing, and generate help information.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "entrypoints", metavar="entrypoint", nargs="+", help="python file defining protocols/algorithms/etc.", type=str
    )
    parser.add_argument("-c", "--config", help="config file", type=str, required=True)
    parser.add_argument("--list-protocols", help="Print the available protocols", action="store_true")
    parser.add_argument("--list-algorithms", help="Print the available algorithms", action="store_true")
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

    # Load the entrypoints.
    for ep in args.entrypoints:
        try:
            import_source(ep)
        except FileNotFoundError as e:
            logging.error(e)
            return 1

    # Get the list of Tinker protocols and Tinker algorithms.
    protocols = protocol.Protocol.get_impls(subclasses=True)
    algorithms = algorithm.Algorithm.get_impls(subclasses=True)

    # Print out available protocols/algorithms if requested.
    if args.list_protocols or args.list_algorithms:
        if args.list_protocols:
            print_objects(protocols, "Protocols")

        if args.list_algorithms:
            print_objects(algorithms, "Algorithms")

        return 0

    # If there is a single protocol to run, then instantiate it and run it.
    if len(protocols) == 1:
        protocol_cls = protocols.pop()
        p = protocol_cls()

        try:
            p.run_protocol()
        except BaseException:
            exc = sys.exc_info()[1]
            logging.error(f"Protocol runtime error: {exc}")
            return 1
    else:
        logging.error("Fatal error: no protocol specified")

    return 0


if __name__ == "__main__":
    main()
