# -*- coding: utf-8 -*-
"""
This script clusters a set of traces and then chooses a few tests per cluster.

The resulting regression test suite is saved in the file regression_tests.json.

Created on Thu Mar 26 2020.

@author: m.utting@uq.edu.au
"""

from pathlib import Path
import sys

import agilkia

# %% Global defaults

TESTS_PER_CLUSTER = 3
OUTPUT_FILE = "regression_tests.json"

# %% cluster traces, select, and return the new TraceSet.

def cluster_select(traces: agilkia.TraceSet, num_tests: int):
    data = traces.get_trace_data(method="action_counts")
    print(data.sum().sort_values())
    num_clusters = traces.create_clusters(data)
    print(f"{num_clusters} clusters found.  Selecting first {num_tests} / cluster.")
    result = agilkia.TraceSet([], meta_data=traces.meta_data)
    for i in range(num_clusters):
        print(f"Cluster {i}:")
        for tr in traces.get_cluster(i)[0:num_tests]:
            print(f"    {tr}")
            result.append(tr)
    return result


# %% Do it on the given file.

def main(args):
    if len(args) >= 2:
        want = TESTS_PER_CLUSTER
        start = 1
        if args[1].startswith("--tests-per-cluster="):
            want = int(args[1].split("=")[1])
            start += 1
        traces = None
        for name in args[start:]:
            if traces is None:
                traces = agilkia.TraceSet.load_from_json(Path(name))
            else:
                traces.extend(agilkia.TraceSet.load_from_json(Path(name)).traces)
        result = cluster_select(traces, want)
        result.save_to_json(Path(OUTPUT_FILE))
    else:
        script = sys.argv[0] or "choose_regression_tests.py"
        print(f"This script reads traces in Agilkia *.json trace format, clusters them, then")
        print(f"selects a given NUM (default 3) of tests per cluster for regression testing.")
        print(f"The resulting test suite is saved into '{OUTPUT_FILE}'.")
        print(f"Setup: conda install -c mark.utting agilkia")
        print(f"Usage: python {script} [--tests-per-cluster=NUM] traces.json ...")

# %%

if __name__ == "__main__":
    main(sys.argv)
