# Tinker examples using NetHarn
Examples ported from NetHarn: https://gitlab.kitware.com/computer-vision/netharn

# Install NetHarn dependencies:
pip install -r examples/netharn/requirements.netharn.txt 

# Create a blank configuration.json for tinker
touch examples/configuration.json

# Run NetHarn MNIST classifier with tinker:
See: https://gitlab.kitware.com/computer-vision/netharn/-/blob/master/netharn/examples/mnist.py

tinker examples/netharn/mnistclassify.py -c examples/configuration.json

