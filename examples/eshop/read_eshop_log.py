# -*- coding: utf-8 -*-
"""
Example project to read e-Shop logs and save in Agilkia JSON format.

It can also split the traces into sessions based on the SessionID.

It uses the Philae 'Agilkia' library, which can be installed by::

    conda install -c mark.utting agilkia

Created April 2021

@author: m.utting@uq.edu.au
"""

import json
from pathlib import Path
from datetime import datetime
import sys

import agilkia

# %%

def read_traces_log(path: Path) -> agilkia.TraceSet:
    """Reads the given log file as a single long trace.

    Note that the sequence id number and datetime stamp are recorded as meta data of each event,
    with the datatime stamp converted to a Python datetime object.

    The "sessionID" and date-time is recorded as meta-data of the event.
    The "function" is used as the action name.
    The "customerID" and "controller" are treated as inputs.
    The "httpResponseCode" is the output and is renamed to 'Status'.

    For example, theis line of an input file::

      2021-03-04 10:19:58 - {"sessionID":"0fb311c015e861eebebd596d90","customerID":0,"controller":"ControllerCommonHome",
        "function":"index","data":{"route":"common\/home"},"httpResponseCode":200}
    
    will translate into this Event::

        Event("index", inputs1, outputs1, meta1)
        where
          inputs1 = {"customerID": "0", "controller": "ControllerCommonHome"}
          outputs1 = {"Status": 200}
          meta1 = {"sessionID": 0fb311c015e861eebebd596d90, "timestamp": <2021-03-04T10:19:58>}
    """
    # print("now=", datetime.now().timestamp())
    with path.open("r") as input:
        trace1 = agilkia.Trace([])
        for line in input:
            sep = line.find(" - ")
            if sep <= 0:
                print("WARNING: skipping line:", line)
                continue

            timestamp = datetime.strptime(line[0:sep], '%Y-%m-%d %H:%M:%S')
            contents = json.loads(line[sep+3:])
            # print(timestamp, contents)

            sessionID = contents["sessionID"]
            customerID = contents["customerID"]
            controller = contents["controller"]
            function = contents["function"]
            data = contents["data"]
            response = contents["httpResponseCode"]
            # now we identify inputs, outputs, and meta-data.
            inputs = {
                    'customerID': customerID,
                    'controller': controller,
                    }
            inputs.update(data)  # add all the data keys and values.
            outputs = {'Status': int(response)}
            meta = {
                    'sessionID': sessionID,
                    'timestamp': timestamp
                    }
            event = agilkia.Event(function, inputs, outputs, meta)
            trace1.append(event)
            # See which data keys are associated with each type of event?
            # print(function, sorted(list(data.keys())))
    traceset = agilkia.TraceSet([])
    traceset.append(trace1)
    return traceset


# %% How to visualise traces
# Since there are so many different types of actions, we use two chars for each one.

# We use lowercase or punctuation for common actions, uppercase for less common ones.
abbrev_chars = {
    "add": "+",
    "addAddress": "A",
    "addCustomer": "C",
    "addOrderHistory": "O",
    "addReview": "R",
    "agree": "Y",  # like "Yes"
    "alert": "!",
    "confirm": "y",  # for "yes"
    "country": "c",
    "coupon": "#",
    "currency": "$",
    "customfield": "f",
    "delete": "d",
    "deleteAddress": "D",
    "edit": "e",
    "editPassword": "E",
    "getRecurringDescription": "X",
    "index": ".",
    "login": "L",
    "quote": "q",
    "remove": "x",
    "reorder": "&",
    "review": "r",
    "save": "s",
    "send": "S",
    "shipping": "^",
    "success": "=",
    "voucher": "v",
    "write": "w"
    }

# check that the abbrev chars are all unique.
assert len(abbrev_chars.keys()) == len(abbrev_chars.values())

# %% Read traces and save in the Agilkia JSON format.

def read_split_save(name: str, split: bool, verbose: bool = False):
        path = Path(name)
        traces = read_traces_log(path)
        traces.set_event_chars(abbrev_chars)
        msg = ""
        if split:
            traces = traces.with_traces_grouped_by(key=(lambda ev: ev.meta_data["sessionID"]))
            path2 = path.with_suffix(".split.json")
        else:
            path2 = path.with_suffix(".json")
        if verbose:
            print("# abbrev_chars =", abbrev_chars)
            for tr in traces:
                print(tr)
        print(f"  {path} -> {path2} [{len(traces)} traces{msg}]")
        traces.save_to_json(path2)

# %%

def main(args):
    start = 1
    split = False
    verbose = False
    if start < len(args) and args[start] == "--split":
        start += 1
        split = True
    if start < len(args) and args[start] == "--verbose":
        start += 1
        verbose = True
    files = args[start:]
    if len(files) > 0:
        for name in files:
            read_split_save(name, split=split, verbose=verbose)
    else:
        script = args[0] or "read_eshop_log.py"
        print("This script converts e-shop log files into Agilkia *.json trace files.")
        print("If the --split argument is given, it will also split traces by session IDs.")
        print("If the --verbose argument is given, it will print the trace(s).")
        print(f"Setup: conda install -c mark.utting agilkia")
        print(f"Usage: python {script} [--split] [--verbose] log1.txt log2.txt ...")


# %%

if __name__ == "__main__":
    main(sys.argv)

