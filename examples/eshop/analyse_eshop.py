# -*- coding: utf-8 -*-
"""
Example project to read e-Shop logs and print statistics about them.

Input files must be in Agilkia JSON format and should usually be split into sessions.

It uses the Philae 'Agilkia' library, which can be installed by::

    conda install -c mark.utting agilkia

Created April 2021

@author: m.utting@uq.edu.au
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

import agilkia


# %% Print statistics

def undict(val):
    if isinstance(val, dict):
        return "<dict>"
    else:
        return val


def analyse(traceset: agilkia.TraceSet):
    print("# Example of logged events as Pandas table")
    data = traceset.to_pandas()
    print(data)
    print("# Column data descriptions (dict values replaced by 'dict')")
    # describe() does not like dict values, so we replace dict values by just string "dict"
    tmp = data.applymap(undict)
    print(tmp.describe(include='all').transpose())
    average = data.shape[0] / len(traceset)
    print(f"{len(traceset)} traces, with average {average:.1f} events each.")


# %%
def main(args):
    start = 1
    verbose = False
    if start < len(args) and args[start] == "--verbose":
        start += 1
        verbose = True
    files = args[start:]
    if len(files) > 0:
        for name in files:
            traces = agilkia.TraceSet.load_from_json(Path(name))
            analyse(traces)
            if verbose:
                print("# abbrev_chars =", traces.get_event_chars())
                for tr in traces:
                    print("  ", tr)
    else:
        script = args[0] or "analyse_eshop.py"
        print("This script analyses e-shop logs (in Agilkia *.json format) and prints statistics.")
        print("If the --verbose argument is given, it will print all the trace(s).")
        print(f"Setup: conda install -c mark.utting agilkia")
        print(f"Usage: python {script} [--verbose] log1.json")


# %%

if __name__ == "__main__":
    main(sys.argv)
