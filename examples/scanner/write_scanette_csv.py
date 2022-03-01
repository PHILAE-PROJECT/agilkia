# -*- coding: utf-8 -*-
"""
Example project to convert from Agilkia JSON format into Scanette CSV log format.

It uses the Philae 'Agilkia' library, which can be installed by::

    conda install -c mark.utting agilkia

Created on Wed Mar 25 2020

@author: m.utting@ua.edu.au
"""

from pathlib import Path
import sys
import math

import agilkia

# %%

def scanette_status(intval, floatval):
    """Convert Event status value into Scanette status/result field.

    Scanette result values are sometimes int, sometimes float, sometimes "?".
    """
    if intval == floatval:
        return intval
    if math.isnan(floatval):
        return "?"
    return floatval


def write_traces_csv(traces: agilkia.TraceSet, path: Path):
    """Writes the given traces into a CSV file.

    Note that the sequence id number and datetime stamp are recorded as meta data of each event,
    with the datatime stamp converted to a Python datetime object.

    The "sessionID", and "object" instance name are recorded as inputs of the event.
    The optional parameter is also added to the inputs under the name "param", if present.

    For example, these two lines of an input CSV file::

        203, 1584454658227, client9, caisse1, fermerSession, [], 0
        208, 1584454658243, client9, caisse1, payer, [260], 8.67
    """
    # print("now=", datetime.now().timestamp())
    time = 1585034888279  # 24 Mar 2020.
    with path.open("w") as output:
        n = 0
        for tr in traces:
            for ev in tr:
                n += 1
                if "timestamp" in ev.meta_data:
                    time = int(ev.meta_data["timestamp"].timestamp() * 1000)
                if "param" in ev.inputs:
                    params = f"[{ev.inputs['param']}]"
                else:
                    params = "[]"
                sess = ev.inputs["sessionID"]
                obj = ev.inputs["object"]
                status = scanette_status(ev.status, ev.status_float)
                output.write(f"{n}, {time}, {sess}, {obj}, {ev.action}, {params}, {status}\n")


# %% Read traces and save in the Agilkia JSON format.

def convert_to_csv(name: str):
        path = Path(name)
        traces = agilkia.TraceSet.load_from_json(path)
        path2 = path.with_suffix(".csv")
        if path2.exists():
            raise Exception(f"Output file {path2} exists.  Please remove it first.")
        print(f"  {path} -> {path2} [{len(traces)} traces]")
        write_traces_csv(traces, path2)

# %%

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        for name in sys.argv[1:]:
            convert_to_csv(name)
    else:
        script = sys.argv[0] or "write_scanette_csv.py"
        print("This script converts Agilkia *.json trace files into Scanette *.csv files.")
        print(f"Setup: conda install -c mark.utting agilkia")
        print(f"Usage: python {script} traces1.json traces2.json ...")

