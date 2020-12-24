"""
The `tinker` CLI utility.

For command line documentation, run `tinker --help`.
"""

import argparse
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


def main() -> int:
    """Run the main program."""

    # Setup the argument parsing, and generate help information.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "protocol_files",
        metavar="protocol_file",
        nargs="+",
        help="python file defining protocols/algorithms/etc.",
        type=str,
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

    # Load the protocol files.
    for pf in args.protocol_files:
        try:
            import_source(pf)
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
        return 1

    return 0


if __name__ == "__main__":
    main()
