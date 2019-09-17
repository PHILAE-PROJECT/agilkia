"""Automated smart testing strategies for web services."""

# This package follows a 'Convenience Store' model.
# That is, it directly exports all the features that will be useful to users.
# They do not need to import sub-modules.
#
# See the article: "What’s __init__ for me?" by Jacob Deppen on TowardsDataScience.com:
# https://towardsdatascience.com/whats-init-for-me-d70a312da583

__version__ = '0.1'

from . random_tester import RandomTester, uniq, build_interface, print_signatures, 
    DUMP_WSDL, DUMP_SIGNATURES
from . json_traces import MyEncoder
