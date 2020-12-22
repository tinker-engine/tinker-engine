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
import click
import importlib.util
import logging
import os
import pkg_resources
from pkg_resources import EntryPoint
import smqtk  # type: ignore
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
    """Print out `objects` in a human-readable report."""
    print(f"{title}:")
    if not objects:
        print("  none found")
    else:
        for o in objects:
            print(f"  {o.__module__}.{o.__name__}")


discovered_plugins = {
    entry_point.name: _safe_load(entry_point) for entry_point in pkg_resources.iter_entry_points("tinker")
}


@click.command()
@click.option("-c", "--config", required=True, type=click.Path(exists=True), help="config file")
@click.option("--list-protocols", is_flag=True, help="Print the available protocols")
@click.option("--list-algorithms", is_flag=True, help="Print the available algorithms")
@click.option("--log-file", default=f"tinker_{socket.gethostname()}_{time.asctime().replace(' ', '_')}.log", help="Path to log file")
@click.option("--log-level", default=logging.INFO, type=int, help="Logging level")
@click.argument("protocol-files", type=click.Path(exists=True), nargs=-1, required=True)
def main(config, list_protocols, list_algorithms, log_file, log_level, protocol_files) -> int:
    """Run computational experiments via custom configuration and protocols.

    PROTOCOL_FILES is one or more Python files defining protocols.
    """

    # Set up the logger.
    log_format = "[tinker-engine] %(asctime)s %(message)s"
    logging.basicConfig(filename=log_file, filemode="w", level=log_level, format=log_format)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # Find the config file.
    if not os.path.exists(config):
        logging.error(f"error: config file {config} doesn't exist")
        return 1

    # Load the protocol files.
    for pf in protocol_files:
        try:
            import_source(pf)
        except FileNotFoundError as e:
            logging.error(e)
            return 1

    # Get the list of Tinker protocols and Tinker algorithms.
    protocols = protocol.Protocol.get_impls(subclasses=True)
    algorithms = algorithm.Algorithm.get_impls(subclasses=True)

    # Print out available protocols/algorithms if requested.
    if list_protocols or list_algorithms:
        if list_protocols:
            print_objects(protocols, "Protocols")

        if list_algorithms:
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
        return 1

    return 0


if __name__ == "__main__":
    main()
