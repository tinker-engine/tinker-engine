import abc
import os
import sys
import inspect

import pkg_resources
from typing import Any, Optional, TypeVar
from pkg_resources import EntryPoint

def _safe_load(entry_point: EntryPoint) -> Optional[Any]:
    """Load algorithms from an entrypoint without raising exceptions."""
    try:
        return entry_point.load()
    except Exception as fault:
        print("Cannot load entrypoint")
        print( fault )
        exit(1)



discovered_plugins = {
    entry_point.name: _safe_load(entry_point)
    for entry_point in pkg_resources.iter_entry_points("framework")
}


class BaseProtocol(metaclass=abc.ABCMeta):
    """ The BaseProtocol class provides a generic toolset storage and a mechanism to
        retreive algorithms given their filename.
    """
    def __init__(self,algodirectory, harness):
        self.test_harness = harness
        self.algorithmsbase = algodirectory
        self.toolset = dict()

    @abc.abstractmethod
    def run_protocol(self):
        raise NotImplementedError

    def get_algorithm(self, algotype, toolset):
        ''' get_algorithm loads a single algorithm file from the filesystem and instantiates a single
            object of the relevant type from that file.

            Arguments:
                algotype: (option 1) This is a string that contains either an absolute or relative path to the
                            desired algorithm file. If the path is relative, then the location of
                            the file is determined using the self.algorithmsbase directory and
                            appending the algotype string to it to generate the absolute file path.

                algotype: (option 2) This is a string that names a plugin that has been pip installed. This
                            function will attempt to load using the plugin if a file with a matching name to
                            algotype can't be found.

                toolset: This is a dict() containing the initialization information for the algorithm.

            Returns:
                a single instantiated object of the desired algorithm class.


        '''
        # load the algorithm from the self.algorithmsbase/algotype
        # TODO: implement the mechanism to override the normal behavior of this function and use it to create
        #       template algorithm files and adapters instead.

        # Validate the toolset is a dictionary or None
        if toolset:
            if not isinstance( toolset, dict):
                print( "toolset must be a dictionary")
                exit(1)


        # if the file exists, then load the algo from the file. if not, then load the algo from plugin
        algofile = os.path.join(self.algorithmsbase, algotype)
        if os.path.exists(algofile) and not os.path.isdir(algofile):
            print(algotype, "found in algorithms path, loading file")
            return self.load_from_file(algofile,toolset)
        else:
            print(algotype, "not found in path, loading plugin")
            return self.load_from_plugin(algotype, toolset)

    def load_from_file( self, algofile, toolset):

        # get the path to the algo file so that we can append it to the system path
        # this makes the import easier
        argpath, argfile = os.path.split(algofile)
        if argpath:
            sys.path.append(argpath)

        # load the file as a module and create an object of the class type in the file
        # the name of the class doesnt matter, as long as there is only one class in
        # the file.
        argbase, argext = os.path.splitext(argfile)
        if argext == ".py":
            algoimport = __import__(argbase, globals(), locals(), [], 0)
            for name, obj in inspect.getmembers(algoimport):
                if inspect.isclass(obj):
                    foo = inspect.getmodule(obj)
                    if foo == algoimport:
                        #construct the algorithm object
                        algorithm = obj(toolset)
        else:
            print("Given algorithm is not a python file, other types not supported")
            exit(1)

        return algorithm


    def load_from_plugin( self, algotype, toolset):
        algorithm = discovered_plugins.get(algotype)
        if algorithm is None:
            print("Requested plugin not found")
            exit(1)
        return algorithm(toolset)

