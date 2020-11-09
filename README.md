# tinker-engine

## Developer setup

### Build Tinker Enginer

The easiest way to build Tinker Engine locally is by using Pipenv. These steps
will create a Pipenv environment where Tinker Engine is built and installed and
available for use.

1. Clone this repository.
2. Clone the [SMQTK
   repository](https://kwgitlab.kitware.com/computer-vision/SMQTK/), making sure
   to check out the `prr-diyai-20201015` tag (i.e., `git clone prr-diyai-20201015 -b prr-diyai-20201015`).
3. In the `tinker-engine` repository, install the Pipenv dependencies with
   `pipenv install -d`.
4. Install the SMQTK dependency manually with `pipenv run pip install -e
   ../smqtk` (substituting your local path to SMQTK).
5. Activate the Pipenv shell: `pipenv shell`.
6. Run Tinker Engine to ensure it works: `tinker --help`.

### Running Tinker Engine

There is an example protocol definition in
[`examples/helloworld.py`](examples/helloworld.py). You can use this as a Tinker
Engine entrypoint as follows:

```
$ tinker -c <any file that exists> examples/helloworld.py
```

(Currently, the config file argument is required, but isn't actually used by the
system; hence, you must supply a file that exists in order for this to work.)

Since Tinker Engine only finds a single protocol defined in the entrypoints
supplied to it, it will automatically instantiate and run the one it has found.
But you can also list the ones it knows about, like this:

```
$ tinker -c examples/helloworld.yaml examples/helloworld.py --list-protocols
```

# IGNORE EVERYTHING BELOW

## Getting started

This repository contains a development version of the `tinker-engine` package.
To install it, make sure you are in your desired Python environment (e.g., by
creating a virtual environment), enter the top level directory of this
repository, and use `pip`:

```bash
pip install -e .
```

Once this is complete, you can execute protocols using Tinker Engine using the following command
```bash
tinker protocol_file
```
protocol\_file is the location and name of a python protocol file. The project administrator should provide the protocol script to the researchers for their local development. The protocol\_file can be either an absolute file path, or a path from the current working directory to the protocol file. Tinker Engine can be executed from any location, but it is recomended that it be run from the root of the algorithms directory.

If Tinker Engine needs to be run from somewhere other than the algorithms directory root, the path to the algorithms direcotry root should be provided using the -a argument

```bash
tinker protocol_file -a alogorithms_path
```
 
Tinker Engine can also be run with different harness interfaces. Use the -l flag to list the available interfaces. By default Tinker Engine uses the LocalInterface provided by Tinker Engine. An alternate can be selected using the -i \<interface name\> argument. User defined interfaces can also be used, but must derive from the Harness base class, and should be placed in the same folder with the protocol file. If everything is setup correctly, the -l flag will also print out the name of any user defined interfaces that are available.

## Running Learn

First, install Tinker Engine (instructions above). Next, check out the Learn
repository. Use the `pip` command to install the
Learn package: if the `tinker-engine` and `learn` repositories are in side-by-side
directories, and you are in the top-level directory of `tinker-engine`, then the
appropriate command would be `pip install -e ../learn`.

Then you will need to download all of the datasets and follow
instructions from: https://gitlab.lollllz.com/lwll/dataset_prep. Make sure the
`lwll_datasets` directory is in your current directory.

To run the learn protocol, invoke the `tinker` command as follows:
```bash
tinker ../learn/learn/protocol/learn_protocol.py -a ../learn/learn/algorithms
```

## System design
Tinker Engine has three parts. Tinker Engine itself is installed such that it can be executed from any desired location
on the system. The second part is the protocol and optional interfaces. The protocol is written by the project team to meet the requirements of
the entire project. This is a python script that exercises all parts of the system in the desired fashion. The project team may wish to also include simpler protocols that excercise just one algorithm at a time. This is useful for researchers when developing algorithms. Additionally, if the standard local interface test harness is not sufficient for the project needs, a substitute interface can be defined by the project team, as long as it inherits from the Harness class. The third part are the algorithms. The three parts of the system do not need to be in the same location. The only requirement for the algorithms is that they all reside under the same root folder, and any user defined interfaces must be in the same folder with the protocol file.

The basic design of the system is in support of a two tier system of users. The first tier are the project administrators. At this level, the experimental protocol is established and codified. To make this job easier, the final protocol code is written in python, and has very little restrictions to how it works. Tinker Engine itself mostly provides a support structure for the user written protocol file. The system diagram for this is:

![Tinker Engine System Diagram](images/tinker engine.png)

Under this framework, the start of execution begins in the main.py, which locates, prepares and loads the protocol from the given protocol python file. It also loads the requested interface and supplies that to the protocol. The name of the file is provided by the end-user thorugh a command line argument, and the name of the class within the protocol file is unimportant (it will be loaded regardless of its name). Execution is then passed to the protocols run\_protocol function. The user protocol should inherit from tinker.BaseProtocol. the BaseProtocol provides access to the get\_algorithm function which will locate and dynamically load an algorithm on request. When the protocol needs to pass execution to a specific algorithm, it should call that algorithms "execute" function. One of the arguments to that function provides a step\_description which indicates what mode of processing the algorrithm should perform. In order to facilitate this behavior, every algorithm class must derive from tinker.BaseAlgorithm directly, or from an adapter class that derives from BaseAlgorithm. Like the protocol files, the name of the class in an algorithm file is ignored, and the class is loaded regardless of its name.

When implementing an algorithm, a "requirements.txt" file must be provided so that we can figure
out the algorithm's dependencies. An example dependencies file looks something like:

```
torch>=1.3.0
torchvision>=0.3.0
numpy>=1.16.4
scikit-learn>=0.21.1
scipy>=1.2.1
ubelt>=0.8.7
```

## Acknowledgement of Support and Disclaimer

This material is based upon work supported by the Defense Advanced Research
Projects Agency (DARPA) under Contract No. HR001120C0055. Any opinions, findings
and conclusions or recommendations expressed in this material are those of the
author(s) and do not necessarily reflect the views of the DARPA.  

This material is based on research sponsored by DARPA under agreement number
FA8750-19-1-0504. The U.S. Government is authorized to reproduce and distribute
reprints for Governmental purposes notwithstanding any copyright notation
thereon.  
