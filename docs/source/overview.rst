.. _overview:

========
Overview
========

Tinker Engine is a Python framework for designing experimental *protocols*
featuring specific ways to run *algorithms*, supplying each run with *input
data*, and capturing *output data*, all under the control of one or more
*configurations*. Tinker Engine orchestrates these pieces under your ultimate
control, so that you can learn the performance characteristics of the algorithms
and protocols you are studying.

This section gives an overview of Tinker Engine, what it is, and how it works.

Computational Experiments
-------------------------

In many computational situations, you want to evaluate how well different
solutions perform on a given problem. Generally, you will follow a scientific
comparison process to find out, perhaps by running each of several algorithms on
an identically prepared setup. Further, you may have different such setups, or
different ways in which the algorithms are invoked, etc.

We refer to such situations as *computational experiments*, and Tinker Engine is
designed to give you the power to describe the algorithms, the protocols under
which they are invoked, and the manner in which batches of algorithms are to be
compared with one another.

Tinker Engine Example
---------------------

Tinker Engine Abstractions
--------------------------
