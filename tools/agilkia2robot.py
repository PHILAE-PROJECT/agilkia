# -*- coding: utf-8 -*-
"""
Output an Agilkia TraceSet as a sequence of RobotFramework tests.

The output file will have the same name as the input file, but with the extension .robot.
"""
# @author: Mark Utting, m.utting@uq.edu.au

import argparse
from pathlib import Path
from typing import List
import agilkia

INDENT = "    "
SEP = "    "   # separator between Robot parameters


def output_line(values: List[str], output):
    """Outputs a sequence of strings as an indented RobotFramework line."""
    line = SEP.join(values)
    output.write(f"{INDENT}{line}\n")


def output_event(event: agilkia.Event, output):
    """Writes the event as two lines, send inputs, then check outputs."""
    output_line([event.action] + [str(event.inputs[k]) for k in event.inputs], output)
    output_line(["check status", str(event.status)], output)


def robot_output(traceset: agilkia.TraceSet, output):
    output.write("*** Test Cases ***\n")
    for i, tr in enumerate(traceset):
        output.write(f"Test {i}\n")
        for ev in tr:
            output_event(ev, output)


def main():
    """A command line program that prints a set of traces as RobotFramework tests."""
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser.add_argument("-e", "--eventchars", help="a csv file containing an event-to-char map.")
    # parser.add_argument("-n", "--noclusters", help="view traces in-order, without clusters.",
    #     action="store_true")
    parser.add_argument("traceset", help="an Agilkia traceset file (*.json)")
    args = parser.parse_args()
    # print(f"Args are:", args)
    infile = Path(args.traceset)
    traceset = agilkia.TraceSet.load_from_json(infile)
    if infile.suffixes[-3:] == [".agilkia", ".json", ".gz"]:
        # remove final two suffixes
        infile = infile.with_suffix("").with_suffix("")
    elif infile.suffixes[-2:] == [".agilkia", ".json"]:
        # remove one final suffix
        infile = infile.with_suffix("")
    out_file = Path(infile).with_suffix(".robot")
    with open(out_file, "w") as output:
        robot_output(traceset, output)


if __name__ == "__main__":
    main()
