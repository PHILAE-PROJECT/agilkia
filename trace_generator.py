# -*- coding: utf-8 -*-
"""
Generate and execute tests for a Web Service.

Example args: http://www.soapclient.com/xml  soapresponder.wsdl
@author: utting@usc.edu.au
"""

import random
import pandas as pd
import argparse
from pathlib import Path

import agilkia

WSDL_EG = "http://www.soapclient.com/xml"
WS_EG = "soapresponder"


def main():
    """A command line program to generate and execute test traces for a SOAP web service."""
    parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument("-c", "--compress", help="compress repeats of this action")
    parser.add_argument("-n", "--tests", type=int, default=2, help="number of tests to generate")
    parser.add_argument("-l", "--length", type=int, default=5, help="length of each test")
    parser.add_argument("-i", "--inputs", help="input values CSV file: (Input,Freq,Value) tuples")
    parser.add_argument("-a", "--actionchars", help="name of action-to-char mapping file (*.csv)")
    parser.add_argument("-m", "--methods", help="methods to test (comma-separated)")
    parser.add_argument("-v", "--verbose", default="true",
                        help="print each test after execution", action="store_true")
    parser.add_argument("-s", "--seed", type=int, help="seed for random generator")
    parser.add_argument("-o", "--output", type=str, default="out.json", help="name of output file")
    parser.add_argument("url", help="URL of web service server")
    parser.add_argument("service", nargs='*', help="name of a web service")
    args = parser.parse_args()
    # print(f"Args are:", args)

    json_output = Path(args.output).with_suffix(".json")
    if json_output.exists():
        raise Exception(f"do not want to overwrite {json_output}.")
    # process arguments
    action_chars = None
    if args.map:
        mapfile = pd.read_csv(args.map, header=None)
        # we assume this has just two columns: 0=action_name and 1=char.
        action_chars = dict(zip(mapfile.iloc[:, 0], mapfile.iloc[:, 1]))
    input_rules = None
    if args.inputs:
        input_rules = agilkia.read_input_rules(Path(args.inputs))
    methods = None
    if args.methods:
        methods = args.methods.split(",")
    rand = random.Random(args.seed)  # seed can be None
    print(f"Starting to test {args.url} with services: {args.service}")
    tester = agilkia.RandomTester(args.url, args.service, rand=rand, verbose=args.verbose,
                                  action_chars=action_chars, input_rules=input_rules,
                                  methods_to_test=methods)
    # TODO: methods_to_test=None,
    for i in range(args.tests):
        trace = tester.generate_trace(length=args.length, start=True)
        if args.verbose:
            print(f"  {str(trace)}")
    tester.trace_set.save_to_json(json_output)


if __name__ == "__main__":
    main()
