# -*- coding: utf-8 -*-
"""
This script clusters a set of traces and then chooses a few tests per cluster.

The resulting regression test suite is saved in the file regression_tests.json.

Created on Thu Mar 26 2020.

@author: m.utting@uq.edu.au
"""

from pathlib import Path
from random import Random
import sys

import agilkia

# %% Global defaults

TESTS_PER_CLUSTER = 3
INPUT_FILE = "1026-steps.split.json"
OUTPUT_FILE = "regression_tests.json"

# %% cluster traces, select, and return the new TraceSet.

def cluster_select(traces: agilkia.TraceSet, num_tests: int, rand=None) -> agilkia.TraceSet:
    """
    Clusters the given traces, then chooses up to `num_tests` per cluster.

    Parameters
    ----------
    traces : agilkia.TraceSet
        Input set of traces.
    num_tests : int
        The desired number of tests per cluster.  For clusters smaller than this,
        all tests will be selected.
    rand : random.Random
        If not None, then this random generator will be used to select a random
        subset of up to `num_tests` from each cluster.  If None, then the first
        `num_tests` will be selected.

    Returns
    -------
    result : agilkia.TraceSet
        A traceset containing the selected tests.
    """
    data = traces.get_trace_data(method="action_counts")
    print(data.sum().sort_values())
    num_clusters = traces.create_clusters(data)
    chooser = "first" if rand is None else "random"
    print(f"{num_clusters} clusters found.  Selecting {chooser} {num_tests} / cluster.")
    result = agilkia.TraceSet([], meta_data=traces.meta_data)
    for i in range(num_clusters):
        print(f"Cluster {i}:")
        cluster = traces.get_cluster(i)
        if rand is None or len(cluster) <= num_tests:
            chosen = cluster[0:num_tests]
        else:
            chosen = rand.sample(cluster, num_tests)
        for tr in chosen:
            print(f"    {tr[0].inputs.get('sessionID', '???'):10s} {tr}")
            result.append(tr)
    return result


# %% Do it on the given file.

def main(args):
    if len(args) >= 2:
        rand = None
        want = TESTS_PER_CLUSTER
        start = 1
        if args[start].startswith("--tests-per-cluster="):
            want = int(args[start].split("=")[1])
            start += 1
        if args[start].startswith("--random-seed="):
            rand = Random(int(args[start].split("=")[1]))
            start += 1
        traces = None
        for name in args[start:]:
            if traces is None:
                traces = agilkia.TraceSet.load_from_json(Path(name))
            else:
                traces.extend(agilkia.TraceSet.load_from_json(Path(name)).traces)
        result = cluster_select(traces, want, rand=rand)
        result.save_to_json(Path(OUTPUT_FILE))
    else:
        script = sys.argv[0] or "choose_regression_tests.py"
        print(f"This script reads traces in Agilkia *.json trace format, clusters them, then")
        print(f"selects a given NUM (default 3) of tests per cluster for regression testing.")
        print(f"The resulting test suite is saved into '{OUTPUT_FILE}'.")
        print(f"Setup: conda install -c mark.utting agilkia")
        print(f"Usage: python {script} [--tests-per-cluster=NUM] [--random-seed=MMM] traces.json ...")

# %%

if __name__ == "__main__":
    main(sys.argv)
