# Algorithm

The learn framework is separated into two different parts. The `algorithm` folder contains all the code for you to run
your code and interface with the JPL API and the `learn_framework` contains the code to dockerize the files and download
the datasets.  

The `algorithm` folder contains all the code for you to run your algorithm.  This is a standalone folder and everything 
can be run from this directory.  Take a look at the `main.py` file to start the code.  You can run the example using
the following command:  
```python
python main.py
```  
There are four main files to look at here.  The only one you should be editing is the `algorithm.py`.  Look into the 
code/documentation for more information on the code.  
- `main.py`: runs the experiment.  
    - Picks and initializes the problem
    - Initializes the classes for the dataset and algorithm
    - Runs the loops for the train and adapt stages.
- `problem.py`: problem (LwLL) class definition
    - Contains the problem/task information
    - Interacts with the JPL API
- `dataset.py`: Dataset classes
    - Contains the classes for the train and eval datasets fore each stage.  
    - The dataset contain both unlabeled/labeled data
    - Connects with problem class to update the dataset based on the indices of the data.
- `algorithm.py`: algorithm class
    - This is the class you will be editing
    - This contains the calls for the train, adapt, and eval stages of your algorithm.

This code implements the [VAAL algorithm](https://github.com/sinhasam/vaal) which has been modified for this problem. 
It is only an active learning approach and doesn't have any adaption mechanism so the train and adapt stages so these 
 two methods are the same though that is just for this example.   