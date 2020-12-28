"""
The `tinker` CLI utility.

For command line documentation, run `tinker --help`.
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

    This function was adapted from the Python 3 recipe for recreating the Python
    2 `import_source()` function, adapted to fit the immediate needs.

    Arguments:
        path: Absolute or relative path to the file of the Python module to import.
    """

    # Extract the name portion of the path.
    basename = os.path.basename(path)
    module_name = f"tinker.{os.path.splitext(basename)[0]}"

    # Get an import spec from the runtime.
    spec = importlib.util.spec_from_file_location(module_name, path)

    # `spec_from_file_location()` returns an Optional[_Loader] but there are
    # some problems with the type definitions (see
    # https://github.com/python/typeshed/issues/2793). This section performs
    # manual typechecking to handle the case of `None`, and then to signal the
    # proper type to the typechecker.
    if spec is None:
        raise RuntimeError(f"{path}: not a valid Python file")

    assert isinstance(spec.loader, importlib.abc.Loader)

    # Perform the actual import.
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


def print_objects(objects: List[smqtk.algorithms.SmqtkAlgorithm], title: str) -> None:
    """Print out `objects` in a human-readable report."""
    print(f"{title}:")
    if not objects:
        print("  none found")
    else:
        for o in objects:
            print(f"  {o.__module__}.{o.__name__}")


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
