# -*- coding: utf-8 -*-
"""
Example project to read Scanette CSV logs and save in Agilkia JSON format.

It can also split the traces into sessions, and optionally cluster them.

It uses the Philae 'Agilkia' library, which can be installed by::

    conda install -c mark.utting agilkia

Reads Scanette CSV files with these columns:
(Column Docs from Frédéric Dadeau, 2019-10-18.)
  0: id is an identifier of the line (some numbers may be missing)
  1: timestamp is in Linux format with three extra digits for milliseconds.
  2: sessionID provides the identifier of the session - a unique ID for each client.
  3: objectInstance is the object instance on which the operation is invoked.
     (that is, which scanner or checkout)
  4: operation is the name of the operation (action).
  5: parameters is a list of the parameter values, or [] if there are no parameters.
  6: result is the status code returned (? means that the operation does
    not return anything - void)

Created on Wed Mar 25 2020

@author: m.utting@ua.edu.au
"""

import csv
from pathlib import Path
from datetime import datetime
import sys

import agilkia

# %%

def read_traces_csv(path: Path) -> agilkia.TraceSet:
    """Reads the given CSV file as a single long trace.

    Note that the sequence id number and datetime stamp are recorded as meta data of each event,
    with the datatime stamp converted to a Python datetime object.

    The "sessionID", and "object" instance name are recorded as inputs of the event.
    The optional parameter is also added to the inputs under the name "param", if present.

    For example, these two lines of an input CSV file::

        203, 1584454658227, client9, caisse1, fermerSession, [], 0
        208, 1584454658243, client9, caisse1, payer, [260], 8.67
    
    will translate into two Events::

        Event("fermerSession", inputs1, outputs1, meta1)
        Event("payer", inputs2, outputs2, meta2)
        where
          inputs1 = {"sessionID": "client9", "object": "caisse1"}
          inputs2 = {"sessionID": "client9", "object": "caisse1", "param": "260"}
          outputs1 = {"Status": 0.0}
          outputs2 = {"Status": 8.67}
          meta1 = {"sequence": 203, "timestamp": <2020-03-18T00:17:38.227>}
          meta1 = {"sequence": 208, "timestamp": <2020-03-18T00:17:38.243>}
    """
    # print("now=", datetime.now().timestamp())
    with path.open("r") as input:
        trace1 = agilkia.Trace([])
        for line in csv.reader(input):
            step = int(line[0].strip())
            timestr = line[1].strip()
            timestamp = datetime.fromtimestamp(int(timestr) / 1000.0)
            # print(step, timestr, timestamp.isoformat())
            sessionID = line[2].strip()
            objInstance = line[3].strip()
            action = line[4].strip()
            paramstr = line[5].strip()
            result = line[6].strip()
            # now we identify the inputs, outputs, and meta-data.
            inputs = {
                    'sessionID': sessionID,
                    'object': objInstance
                    }
            if paramstr != "[]":
                if  paramstr.startswith("[") and paramstr.endswith("]"):
                    paramstr = paramstr[1:-1]
                inputs["param"] = paramstr
            if result == "?":
                outputs = {}
            else:
                outputs = {'Status': float(result)}
            meta = {
                    'sequence': step,
                    'timestamp': timestamp
                    }
            event = agilkia.Event(action, inputs, outputs, meta)
            trace1.append(event)
    traceset = agilkia.TraceSet([])
    traceset.append(trace1)
    return traceset


# %% How to visualise traces (one char per event).

french_chars = {'scanner': '.',
                'abandon': 'a',
                'supprimer': '-',
                'ajouter': '+',
                'debloquer': 'd',
                'fermerSession': 'f',
                'ouvrirSession': 'o',
                'payer': 'p',
                'transmission': 't'
                }

# Our AITest 2020 paper uses these English names and chars.
english_chars = {'scanner': '.',        # 'scan'
                 'abandon': 'a',        # 'abandon'
                 'supprimer': 'd',      # 'delete'
                 'ajouter': '+',        # 'add'
                 'debloquer': 'u',      # 'unlock'
                 'fermerSession': 'c',  # 'closeSession'
                 'ouvrirSession': 'o',  # 'openSession'
                 'payer': 'p',          # 'pay'
                 'transmission': 't'    # 'transmit'
                 }

# %% Read traces and save in the Agilkia JSON format.

def read_split_save(name: str, split: bool, cluster: bool):
        path = Path(name)
        traces = read_traces_csv(path)
        traces.set_event_chars(english_chars)
        msg = ""
        if split:
            traces = traces.with_traces_grouped_by("sessionID")
            if cluster:
                path2 = path.with_suffix(".clustered.json")
                data = traces.get_trace_data()
                num = traces.create_clusters(data)
                msg = f", {num} clusters"
            else:
                path2 = path.with_suffix(".split.json")
        else:
            path2 = path.with_suffix(".json")
        print(f"  {path} -> {path2} [{len(traces)} traces{msg}]")
        traces.save_to_json(path2)

# %%

def main(args):
    start = 1
    split = False
    cluster = False
    if start < len(args) and args[start] == "--split":
        start += 1
        split = True
    if start < len(args) and args[start] == "--cluster":
        start += 1
        cluster = True
    files = args[start:]
    if len(files) > 0:
        for name in files:
            read_split_save(name, split=split, cluster=cluster)
    else:
        script = args[0] or "read_scanette_csv.py"
        print("This script converts Scanette *.csv files into Agilkia *.json trace files.")
        print("If the --split argument is given, it will also split traces by session IDs.")
        print("If --cluster is also given, it will cluster traces using MeanShift with action counts.")
        print(f"Setup: conda install -c mark.utting agilkia")
        print(f"Usage: python {script} [--split [--cluster]] scanner1.csv scanner2.csv ...")


# %%

if __name__ == "__main__":
    main(sys.argv)

