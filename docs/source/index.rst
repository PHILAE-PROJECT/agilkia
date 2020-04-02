.. Agilkia documentation master file, created by
   sphinx-quickstart on Mon Oct 21 15:25:16 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Agilkia!
===================================

Agilkia is a Python toolkit to support AI-for-Testing, especially as applied
to the smart testing of web services and web sites.

The toolkit is intended to make it easier to build testing tools that learn
from traces of customer behaviors, analyze those traces for common patterns
and unusual behaviors (e.g. using clustering techniques), learn machine learning (ML)
models of typical behaviors, and use those models to generate smart tests that
imitate customer behaviors.

Agilkia is intended to provide a storage and interchange format that makes it easy to
built 'smart' tools on top of this toolkit, often with just a few lines of code.
The main focus of this toolkit is saving and loading traces in a standard JSON
format, and transforming those traces to and from lots of other useful formats,
including:

 * Pandas DataFrames (for data analysis and machine learning);
 * ARFF files (for connection to Weka and the StackedTrees tools);
 * SciPy Linkage matrices (for hierarchical clustering and drawing Dendrograms);
 * CSV files in application-specific formats (requires writing some Python code).


.. toctree::
   :maxdepth: 2

   overview
   example
   agilkia


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
