# Tutorials

This directory contains several examples of Tinker Engine protocols along with
associated configurations. This README will explain how to run them and how they
work.

## Coin Flipping

Suppose you want to test out coin flipping algorithms. We'll start out simple,
by implementing a weighted coin simulator, and flipping it a bunch of times
using Tinker Engine.

### Step 1: Model Your Experiment

Start by looking at the [CoinFlipper](coinflip/CoinFlipper.py)
definitions. `CoinFlipper` is an abstract base class that represents the act of
flipping a coin--it returns a `bool`: `true` for heads, `false` for tails. That
file also defines `WeightedCoin`, which is a regular old coin that you're used
to flipping in the ordinary way. A fair coin is a `WeightedCoin` initialized
with `0.5`, while, for example, `WeightedCoin(1.0)` would be a coin with heads
struck on both sides.

You can load this module up and play around with it: go to the
`examples/coinflip` directory, then run `python -i CoinFlipper.py` to enter a
Python shell with `WeightedCoin` defined in the environment. Then create some
coins and play around:

```python
>>> c = WeightedCoin(0.5)
>>> c.flip()
True
>>> c.flip()
False
>>> c.flip()
False
```

### Step 2: Create an Experimental Protocol

Next, we want to set up an experimental protocol. Look at
[coinflip.py](examples/coinflip/coinflip.py). This file defines a Tinker
protocol class whose `run_protocol()` method lays out a simple experimental
protocol: it receives a `config` dictionary, from which it extracts `weight` and
`trials` values; then it constructs a `WeightedCoin` of the given `weight`, then
flips it `trials` times, keeping track of how many heads come up; finally, it
prints a report of the number of heads flipped, together with the expected
number of heads, given the specified coin weighting.

### Step 3: Set Up an Experimental Configuration

The `config` value comes from the [configuration file](coinflip/coinflip.yaml).
Take a look at it--you see that `trials` is set to `1000`, while `weight` has a
more curious structure, reproduced below:

```
weight:
  iterate: [0.0, 0.25, 0.5, 0.75, 1.0]
```

The `iterate` key is a Tinker Engine *directive*; it instructs the system to
treat the value at that point in the configuration as several values to be used
in separate protocol runs. In this case, the `weight` parameter will range over
the values listed. You can see this in action by running the protocol with
Tinker Engine. To do so, ensure you are in the `examples/coinflip` directory,
and then run this command:

```
PYTHONPATH=. tinker -c coinflip.yaml coinflip.py
```

You should see results similar to the following:

```
0 heads out of 1000 flips; expected 0.0
253 heads out of 1000 flips; expected 250.0
508 heads out of 1000 flips; expected 500.0
747 heads out of 1000 flips; expected 750.0
1000 heads out of 1000 flips; expected 1000.0
```

If you look back at the [protocol definition](coinflip/coinflip.py),
you'll recognize that each of the five lines of output corresponds to one run of
the protocol, each with a different value of `config["weight"]` passed in.

### Next Steps

At this point, you can play around with the configuration file and the protocol
to see how things behave. Try replacing the configuration of `trials` with an
`iterate` directive, convert the `iterate` directive for `weight` to a single
value, or vary the contents of the array in the `iterate` directive for
`weight`.
