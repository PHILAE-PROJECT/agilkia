# -*- coding: utf-8 -*-
"""
This script selects a subset of the test set, to be used as a regression test suite.

It can use several alternative optimization algorithms to choose the subset.
The goal is to optimize several coverage metrics, which you can choose.
The chosen coverage metrics will be weighted equally.

The resulting regression test suite is saved in the file regression_tests.agilkia.json.
"""

# @author: Mark Utting, m.utting@uq.edu.au

from pathlib import Path
from random import Random
import argparse
import sys

from agilkia import TraceSet, ObjectiveFunction, EventCoverage, EventPairCoverage, FrequencyCoverage
from agilkia import GreedyOptimizer, ParticleSwarmOptimizer, GeneticOptimizer
from agilkia.trace_set_optimizer import ClusterCoverage


# available coverage metrics
metric_function = {
    "action": EventCoverage(),
    "action_status": EventCoverage(event_to_str=lambda ev: f"{ev.action}_{ev.status}"),
    "action_pair": EventPairCoverage(),
    "frequency": FrequencyCoverage(),
    "cluster": ClusterCoverage()
}


#%% get and check a metric function.

def get_metric(name: str, traces: TraceSet) -> ObjectiveFunction:
    """Translate a name into a coverage metric function, and check it is valid."""
    if name == "frequency" and traces[0].get_meta("freq", None) is None:
        raise Exception("frequency metric needs 'freq' (float) meta_data in each trace.")
    return metric_function[name]


# %% main method to run the chosen optimizer.

def main(args):
    """A command line program that chooses a set of regression tests from a larger set."""
    metric_docs = "Available coverage metrics: " + ", ".join(metric_function.keys())
    parser = argparse.ArgumentParser(description=__doc__ + "\n" + metric_docs,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--metrics", help="the coverage metrics to maximize (comma-separated).",
        default="action_status")
    parser.add_argument("-n", "--number", help="number of tests desired (default=10).",
        type=int, default=10)
    parser.add_argument("-o", "--optimizer", help="algorithm: greedy is faster, pso / genetic optimize better.", 
        choices=["greedy", "pso", "genetic"], default="greedy")
    parser.add_argument("traceset", help="an Agilkia traceset file (*.json)")
    args = parser.parse_args()
    traceset = TraceSet.load_from_json(Path(args.traceset))
    print(f"loaded {len(traceset)} traces, clustered={traceset.is_clustered()}")
    metric_funcs = [get_metric(m, traceset) for m in args.metrics.split(",")]
    if args.optimizer == "greedy":
        optimizer = GreedyOptimizer(metric_funcs)
    elif args.optimizer == "pso":
        optimizer = ParticleSwarmOptimizer(metric_funcs, c1=4)  # you can set extra hyper-params here
    elif args.optimizer == "genetic":
        optimizer = GeneticOptimizer(metric_funcs,)    # you can set extra hyper-params here
    else:
        raise Exception(f"unknown optimizer: {args.optimizer}")
    # run the optimizer
    optimizer.set_data(traceset, args.number)
    tests,coverage = optimizer.optimize()
    print(f"{args.optimizer} optimizer gives {len(tests)} tests, {coverage*100.0:.1f}% coverage")
    tests.save_to_json(Path("regression.agilkia.json"))

# %%

if __name__ == "__main__":
    main(sys.argv)
