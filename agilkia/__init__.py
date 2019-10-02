"""Automated smart testing strategies for web services."""

# This package follows a 'Convenience Store' model.
# That is, it directly exports all the features that will be useful to users.
# They do not need to import sub-modules.
#
# See the article: "What’s __init__ for me?" by Jacob Deppen on TowardsDataScience.com:
# https://towardsdatascience.com/whats-init-for-me-d70a312da583

__version__ = '0.1'

from . random_tester import (RandomTester, uniq, build_interface, print_signatures,
                             DUMP_WSDL, DUMP_SIGNATURES, GOOD_PASSWORD)
from . json_traces import (Trace, TraceSet, TraceEncoder, TRACE_SET_VERSION, xml_decode,
                           all_action_names, event_status,
                           default_map_to_chars, trace_to_string, traces_to_pandas)
