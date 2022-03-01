Agilkia API Docs
================

TraceSet
--------
.. autoclass:: agilkia.TraceSet
   :members:
   :show-inheritance:

Trace
-----
.. autoclass:: agilkia.Trace
   :members:
   :show-inheritance:

Event
-----
.. autoclass:: agilkia.Event
   :members:
   :show-inheritance:

RandomTester
------------
.. autoclass:: agilkia.RandomTester
   :members:
   :show-inheritance:

SmartSequenceGenerator
----------------------
.. autoclass:: agilkia.SmartSequenceGenerator
   :members:
   :show-inheritance:

TracePrefixExtractor
--------------------
.. autoclass:: agilkia.TracePrefixExtractor
   :members:
   :show-inheritance:

TraceEncoder
------------
.. autoclass:: agilkia.TraceEncoder
   :members:
   :show-inheritance:


Agilkia Helper Functions
========================

Most of these global functions are just helper functions for the main agilkia
classes.  However, some may be useful as standalone functions when building
other kinds of tools.


.. module:: agilkia.json_traces

.. autofunction:: xml_decode
.. autofunction:: default_map_to_chars
.. autofunction:: trace_to_string
.. autofunction:: traces_to_pandas


.. module:: agilkia.random_tester

.. autofunction:: read_input_rules
