.. PyPruning documentation master file, created by
   sphinx-quickstart on Tue Jul  6 17:08:02 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyPruning
=========

.. |ls8| image:: ls8-logo-shadow.svg
  :width: 25

.. |sfb| image:: sfb-logo.svg
  :width: 20

.. toctree::
   :maxdepth: 3
   :hidden:

   self
   available
   extending
   papers

This package provides implementations for some common ensemble pruning algorithms. Pruning algorithms aim to select the best subset of an trained ensemble to minimize memory consumption and maximize accuracy. Currently, six types of pruning algorithms are implemented:

- :class:`~PyPruning.RandomPruningClassifier`: Selects a random subset of classifiers. This is mainly used as a baseline.
- :class:`~PyPruning.RankPruningClassifier`: Rank each classifier according to a given metric and then select the best K classifier.
- :class:`~PyPruning.ClusterPruningClassifier`: Cluster the classifiers according to a clustering method and then select a representative from each cluster to from the sub-ensemble.
- :class:`~PyPruning.GreedyPruningClassifier`: Proceeds in rounds and selects the best classifier in each round given the already selected sub-ensemble. 
- :class:`~PyPruning.MIQPPruningClassifier`: Constructs a mixed-integer quadratic problem and optimizes this to compute the best sub ensemble. 
- :class:`~PyPruning.ProxPruningClassifier`: Minimize a (regularized) loss function via (stochastic) proximal gradient descent over the ensembles weights. 

An example on how to use this code can be found in :doc:`available`.

How to install
==============

You can install this package via directly via pip from git 

.. code-block::

   pip install git+https://github.com/sbuschjaeger/PyPruning.git


If you have trouble with dependencies you can try setting up a conda environment which I use for development:

.. code-block::

   git clone git@github.com:sbuschjaeger/PyPruning.git
   cd PyPruning
   conda env create -f environment.yml
   conda activate pypruning

Some notes on the MIQPPruningClassifier
---------------------------------------

For implementing the :class:`~PyPruning.MIQPPruningClassifier` we use `cvxpy <https://www.cvxpy.org/>`_ which does **not** come with a MIQP solver. If you want to use this algorithm you have to manually install a solver, e.g.

.. code-block::

    pip install cvopt

for a free solver or if you want to use a commercial solver and use Anaconda you can also install gurobi (with a free license)

.. code-block::

    conda install -c gurobi gurobi

For more information on setting the solver for :class:`~PyPruning.MIQPPruningClassifier` have a look `here <https://www.cvxpy.org/tutorial/advanced/index.html#solve-method-options>`_.

Acknowledgements
================
The software is written and maintained by `Sebastian Buschjäger <https://sbuschjaeger.github.io/>`_ as part of his work at the `Chair for Artificial Intelligence <https://www-ai.cs.tu-dortmund.de>`_ |ls8| at the TU Dortmund University and the `Collaborative Research Center 876 <https://sfb876.tu-dortmund.de>`_ |sfb|. If you have any question feel free to contact me under sebastian.buschjaeger@tu-dortmund.de.

Special thanks goes to `Henri Petuker <henri.petuker@tu-dortmund.de>`_ who provided parts of this implementation during his bachelor thesis and `David Clemens <david.clemens@tu-dortmund.de>`_ who made the logo. 

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
