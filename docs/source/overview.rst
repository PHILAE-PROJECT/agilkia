Overview
========

Agilkia is a Python library for smart testing of web services and web sites.

It leverages machine learning to analyse and compare customer traces and tests, 
automatically build models of the system under test, and then Generate
tests from those models.  Its goal is to support 'push-button' model-based
testing with a shallow learning curve, but still support more sophisticated
usages that allow experts to optimise test coverage as required.

Key Features
------------

Some of the features that Agilkia supports are:
  * Automated testing of SOAP web services with WSDL descriptions.
  * Manage sets of traces (load/save to JSON, etc.).
  * Convert traces to Pandas DataFrame for data analysis / machine learning.
  * Generate random tests, or 'smart' tests from an ML model.
  * Split traces into smaller traces (sessions).
  * Cluster traces on various criteria, to see common / rare behaviours.
  * Visualise clusters of tests.


Data Structures
---------------

The key data structure in Agilkia is the **TraceSet** class, 
which contains a set of **Trace** objects, which in turn contain a sequence
of **Event** objects.  Each of these three levels can also have *meta-data*
associated with them, such as dates or timestamps, author names, etc.

Here is a little more detail about each of these, starting from the bottom:

  * **Event** (:class:`agilkia.json_traces.Event`) consists of an *action* name,
    a set of named input values, a set of named output values, and some optional
    meta-data such as a 'timestamp'.  If one of the outputs is called 'Status', it
    is assumed to indicate the result status of the whole event, where zero means
    successful and non-zero indicates an error.  If the event also returned a string
    error message, then this should be recorded in an output called 'Error'.  

  * **Trace** (:class:`agilkia.json_traces.Trace`) contains a sequence of Events.
    Each trace knows which TraceSet it is a member of.
    To make traces easy to view, we typically print an abstract view of a trace as
    a string, with just one character per Event.  The mapping from Event action
    names to these single characters is called the 'action_chars' mapping and it
    is stored in the TraceSet so that all traces use the same mapping.

  * **TraceSet** (:class:`agilkia.json_traces.TraceSet`) contains a set of traces,
    plus lots of meta-data.

So the abstract grammar for these data structures is::

    TraceSet ::= MetaData + setof Trace
    Trace    ::= MetaData + setof Event
    Event    ::= MetaData + Action + Inputs + Outputs

where `Action` is a string, and `MetaData`, `Inputs` and `Outputs` are mappings
from strings to any kind of values, such as strings, numbers, datetime objects,
or custom nested objects that were sent to or returned from the system under test.
