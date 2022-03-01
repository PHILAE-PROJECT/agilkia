Overview
========

This 'agilkia' library is part of the Philae research project:

    http://projects.femto-st.fr/philae/en

It is open source software under the MIT license.
See LICENSE.txt

Key Features
------------

* Manage sets of traces (load/save to JSON, etc.).
* Split traces into smaller traces (sessions).
* Cluster traces on various criteria, with support for flat and hierarchical clustering.
* Visualise clusters of tests, to see common / rare behaviours.
* Convert traces to Pandas DataFrame for data analysis / machine learning.
* Generate random tests, or 'smart' tests from an ML model.
* Automated testing of SOAP web services with WSDL descriptions.



Data Structures
---------------

The key data structure in Agilkia is the **TraceSet** class, 
which contains a set of **Trace** objects, which in turn contain a sequence
of **Event** objects.  Each of these three levels can also have *meta-data*
associated with them, such as dates or timestamps, author names, etc.

Here is a little more detail about each of these, starting from the top:

  * **TraceSet** (:class:`agilkia.json_traces.TraceSet`) contains a set of traces,
    plus lots of meta-data, any clustering information about the traces, and the
    `event_chars` mapping that is used to display traces as compact strings.

  * **Trace** (:class:`agilkia.json_traces.Trace`) contains a sequence of Event
    objects.  Each trace knows which TraceSet it is a member of.
    To make traces easy to view, we typically print an abstract view of a trace as
    a string, with just one character per Event.  The mapping from Event action
    names to these single characters is called the `event_chars` mapping and it
    is stored in the TraceSet so that all traces use the same mapping.

  * **Event** (:class:`agilkia.json_traces.Event`) consists of an *action* name,
    a set of named input values, a set of named output values, and some optional
    meta-data such as a 'timestamp'.  If one of the outputs is called 'Status', it
    is assumed to indicate the result status of the whole event, where zero means
    successful and non-zero indicates an error.  If the event also returned a string
    error message, then this should be recorded in an output called 'Error'.

So the abstract grammar for these data structures is::

    TraceSet ::= MetaData + List[Trace]
    Trace    ::= MetaData + List[Event]
    Event    ::= MetaData + Action + Inputs + Outputs

where `Action` is a string, and `MetaData`, `Inputs` and `Outputs` are mappings (dict)
from strings to any kind of values, such as strings, numbers, datetime objects,
or custom nested objects that were sent to or returned from the system under test.
