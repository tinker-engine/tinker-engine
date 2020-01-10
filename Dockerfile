# Start from a cuda base image for enabling GPU execution
FROM pytorch/pytorch

# Copy learn_framework code to the container
COPY ./learn_framework /learn_framework

# Change working directory
WORKDIR /learn_framework

RUN pip install .

# Copy the algorithm folder
COPY ./algorithm /algorithm

WORKDIR /algorithm

# Install algorithms dependencies
RUN pip install -r requirements.txt

# Set main.py file as the entrypoint (executable) of this container
COPY ./entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
