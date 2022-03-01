# -*- coding: utf-8 -*-
"""
Generate and execute tests for a Web Service.

Example args: http://www.soapclient.com/xml/soapresponder.wsdl

@author: m.utting@uq.edu.au
"""

import random
import pandas as pd        # type: ignore
import argparse
from pathlib import Path
from joblib import load    # type: ignore

import agilkia


def main():
    """A command line program to generate and execute test traces for a SOAP web service."""
    parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument("-c", "--compress", help="compress repeats of this action")
    parser.add_argument("-n", "--tests", type=int, default=2, help="number of TESTS to generate")
    parser.add_argument("-l", "--length", type=int, default=5, help="LENGTH of each test")
    fmt = "(CSV with Name,Frequency,Value columns)"
    parser.add_argument("-i", "--inputs", help=f"INPUT values file: {fmt}")
    parser.add_argument("-c", "--chars", help="name of action-to-CHAR mapping file (*.csv)")
    parser.add_argument("-m", "--methods", help="METHODS to test (comma-separated)")
    parser.add_argument("--model", help="ML MODEL (*.joblib) to predict next action.")
    parser.add_argument("-v", "--verbose",
                        help="print VERBOSE messages during testing", action="store_true")
    parser.add_argument("-s", "--seed", type=int, help="SEED for random generator")
    parser.add_argument("-o", "--output", type=str, default="out.json", help="name of OUTPUT file")
    parser.add_argument("url", nargs='+', help="URL of web service server")
    args = parser.parse_args()
    # print(f"Args are:", args)

    json_output = Path(args.output).with_suffix(".json")
    if json_output.exists():
        raise Exception(f"do not want to overwrite {json_output}.")
    # process each optional argument
    action_chars = None
    if args.chars:
        mapfile = pd.read_csv(args.chars, header=None)
        # we assume this has just two columns: 0=action_name and 1=char.
        action_chars = dict(zip(mapfile.iloc[:, 0], mapfile.iloc[:, 1]))
    input_rules = None
    if args.inputs:
        input_rules = agilkia.read_input_rules(Path(args.inputs))
    methods = None
    if args.methods:
        methods = args.methods.split(",")
    rand = random.Random(args.seed)  # seed can be None
    model = None
    if args.model:
        model = load(args.model)
    print(f"Starting to test {args.url}")
    tester = agilkia.RandomTester(args.url, rand=rand, verbose=args.verbose,
                                  action_chars=action_chars, input_rules=input_rules,
                                  methods_to_test=methods)
    # get a password if needed
    if input_rules is not None and "username" in input_rules:
        # this will prompt for a password
        tester.set_username(input_rules["username"])
    # TODO: methods_to_test=None,
    for i in range(args.tests):
        if args.model:
            trace = tester.generate_trace_ml(model, length=args.length, start=True)
        else:
            trace = tester.generate_trace(length=args.length, start=True)
        if args.verbose:
            print(f"  {str(trace)}")
    tester.trace_set.save_to_json(json_output)


if __name__ == "__main__":
    main()
