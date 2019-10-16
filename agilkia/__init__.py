"""Automated smart testing strategies for web services.

This 'agilkia' package is for testing web services and managing set of traces.
Traces may come from user interactions, or from automated test suites, etc.

The main data structure for traces is the ``TraceSet``:
* class TraceSet supports loading/saving traces as JSON, converting to Pandas, etc.
* class Trace is used by TraceSet, and contains a list of Events.
* Each Event is a dict that contains at least the following keys:
    * "action" gives the name of the action (a string);
    * "inputs" is a dict of input parameter names to values;
    * "outputs" is a dict of output parameter names to values.

Automated test generation facilities include:
* RandomTester generates random test sequences.
* SmartTester generates tests from an ML model
  (Currently this is included in RandomTester.generate_trace_ml,
  but this will be split into a separate class shortly.)
"""

# This package follows a 'Convenience Store' model.
# That is, it directly exports all the features that will be useful to users.
# They do not need to import sub-modules.
#
# See the article: "What’s __init__ for me?" by Jacob Deppen on TowardsDataScience.com:
# https://towardsdatascience.com/whats-init-for-me-d70a312da583

__version__ = '0.2.2'

from . random_tester import (read_input_rules, RandomTester,
                             uniq, build_interface, print_signatures,
                             DUMP_WSDL, DUMP_SIGNATURES, GOOD_PASSWORD)
from . json_traces import (Trace, TraceSet, TraceEncoder, TRACE_SET_VERSION, xml_decode,
                           all_action_names, event_status, safe_name,
                           default_map_to_chars, trace_to_string, traces_to_pandas)
