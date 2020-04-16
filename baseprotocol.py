import abc
import os
import sys
import requests
import json
import inspect

class BaseProtocol(metaclass=abc.ABCMeta):
    def __init__(self,algodirectory):
        self.algorithmsbase = algodirectory
        self.toolset = dict()

    @abc.abstractmethod
    def runProtocol(self):
        raise NotImplementedError

    def getAlgorithm(self, algotype):
        # load the algorithm from the self.algorithmsbase/algotype

        #validate that the file exists
        algofile = os.path.join(self.algorithmsbase, algotype)
        if not os.path.exists( algofile ):
            print("given algorithm", algotype, "doesnt exist")
            exit(1)

        #TODO: support handling a directory in the future. The idea would be that the directory
        # would contain only the one algorithm file, and that the protocol wouldn't care what the
        # name of the file was.
        if os.path.isdir( algofile ):
            print("algorithm not yet supported on a directory, use a specific file instead")
            raise NotImplementedError

        # get the path to the algo file so that we can append it to the system path
        # this makes the import easier
        argpath, argfile = os.path.split( algofile )
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
                        algorithm = obj("")
        else:
            print("Given algorithm is not a python file")
            exit(1)

        return algorithm

        
