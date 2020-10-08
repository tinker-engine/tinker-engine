## Train & Eval MNIST classifier tinker example

From the root directory of tinker project:
```
pip install -e .
pip install -r tinker/examples/mnist/requirements.txt

tinker -a tinker/examples/mnist/ -p tinker/examples/mnist/configuration.json tinker/examples/mnist/protocol.py -i MnistHarness
```

