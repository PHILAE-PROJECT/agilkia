# -*- coding: utf-8 -*-
"""
Tool to read CSV files of trace events and convert to Agilkia JSON format.

It can also split the traces into sessions, and optionally cluster them.
Run the tool with no parameters to see detailed help::

    python csv2agilkia.py

Created on 2 March 2022
@author: Mark Utting, m.utting@uq.edu.au
"""

import csv
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Any
import sys

# To install the Agilkia library, do one of::
#   pip install agilkia
#   conda install -c mark.utting agilkia
import agilkia

# %%

def get_column(spec: str, row) -> Optional[Any]:
    """Read a column value from row, and transform it various ways.
    
    A column value is specified by a dot-separated string: N.T1.T2...
    where N is the number of the column to read data from (0...),
    and each of the optional Ti strings is a transformation to apply to that data.

    Transformations are applied left to right.  They include::
     * int: convert the value to an int
     * float: convert the value to a float
     * msec2iso: convert a timestamp (msecs since Jan 1970) to ISO 8601 DateTime string.
     * rmbrackets: remove any '[' ... ']' brackets that surround the string.
     * nonquestion: discard any value equal to '?'.  This will return None instead.
     * nonempty: discard any value equal to ''.  This will return None instead.
    """
    transforms = spec.split(".")
    val = row[int(transforms[0])].strip()
    for f in transforms[1:]:
        if f == "int":
            val = int(val)
        elif f == "float":
            val = float(val)
        elif f == "msec2iso":
            val = datetime.fromtimestamp(int(val) / 1000.0)
        elif f == "rmbrackets":
            # remove optional brackets at beginning+end.
            if val.startswith("[") and val.endswith("]"):
                val = val[1 : -1]
        elif f == "nonquestion":
            if val == "?":
                return None  # immediate return
        elif f == "nonempty":
            if val == "":
                return None  # immediate return
        else:
            raise ValueError(f"unknown column transformation: {f}")
    return val


# tests
row0 = "1584454655792,234, abc , , 56.7".split(",")
assert get_column("1.int", row0) == 234
assert get_column("2", row0) == "abc"
assert get_column("3", row0) == ""
assert get_column("3.nonempty", row0) == None
assert get_column("3.nonempty.int", row0) == None
assert get_column("4", row0) == "56.7"
assert get_column("4.float", row0) == 56.7
assert get_column("0.msec2iso", row0) == datetime.fromisoformat("2020-03-18 00:17:35.792")


def set_field(event: agilkia.Event, field: str, value: Any):
    """Sets one named field of event to the given value.
    
    Example names include: action, status, in.foo, out.bar, meta.zzz.
    """
    if field == "action":
        event.action = value
    elif field == "status":
        event.outputs["Status"] = value
    elif field.startswith("in."):
        event.inputs[field[3:]] = value
    elif field.startswith("out."):
        event.outputs[field[4:]] = value
    elif field.startswith("meta."):
        event.meta_data[field[5:]] = value
    else:
        raise ValueError(f"unknown field: {field}")


def read_traces_csv(path: Path, fields: List[str]) -> agilkia.TraceSet:
    """Reads the given CSV file as a single long trace.

    Each row in the CSV file is assumed to be a single event in a trace,
    which consists of an action, some named input and output fields,
    plus some named meta-data fields such as a session ID, timestamp etc.
    """
    with path.open("r") as input:
        trace1 = agilkia.Trace([])
        for row in csv.reader(input):
            event = agilkia.Event("Unknown", {}, {}, {})
            for f in fields:
                [left, right] = f.split("=")
                val = get_column(right, row)
                if val is not None:
                    set_field(event, left, val)
            # print(f"  adding {event}")
            trace1.append(event)
    traceset = agilkia.TraceSet([])
    traceset.append(trace1)
    return traceset


# %% Read traces and save in the Agilkia JSON format.

def read_split_save(name: str, fields: List[str], split: str, cluster: bool):
    path = Path(name)
    traces = read_traces_csv(path, fields)
    msg = ""
    if split is not None:
        traces = traces.with_traces_grouped_by(split)
        msg = f", grouped by {split}"
        if cluster:
            data = traces.get_trace_data()
            num = traces.create_clusters(data)
            msg = f", {num} clusters"
    path2 = path.with_suffix(".agilkia.json")
    print(f"  {path} -> {path2} [{len(traces)} traces{msg}]")
    traces.save_to_json(path2)

# %%

def main(args):
    start = 1
    splitBy = None
    cluster = False
    if start < len(args) and args[start].startswith("--split="):
        splitBy = args[start].split("=")[1]
        start += 1
    if start < len(args) and args[start] == "--cluster":
        cluster = True
        start += 1
    if start + 1 < len(args):
        fileName = args[start]
        fields = args[start + 1: ]
        assert "action" in [f.split("=")[0] for f in fields]
        read_split_save(fileName, fields, split=splitBy, cluster=cluster)
    else:
        script = args[0] or "csv2agilkia.py"
        print(f"Usage: python {script} [--split=inputName [--cluster]] file.csv field1=column1 field2=column2 ...")
        print()
        print("This script converts CSV files into Agilkia JSON trace files.")
        for s in read_traces_csv.__doc__.split("\n")[2:]:
            print(s)
        print("  --split=COL will group events into traces based on the column named COL.")
        print("  --cluster will cluster traces using MeanShift based on action counts.")
        print()
        print("Each fieldN=columnN specifies how to set an Event field from an input CSV column.")
        print("Valid field names are: action, status, in.I, out.I, meta.I (where I is any identifier).")
        print("For the columnN specifiers:")
        for s in get_column.__doc__.split("\n")[2:]:
            print(s)


# %%

if __name__ == "__main__":
    main(sys.argv)
