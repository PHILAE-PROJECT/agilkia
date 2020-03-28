#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example of creating hierarchical clusters from Scanette traces.

Some helpful documentation about SciPy hierarchical clustering::
    * SciPy docs: https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
    * https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial

@author: m.utting@uq.edu.au
"""

import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hierarchy
from pathlib import Path
import sys

import agilkia

# %% Load Scanette traces

def cluster_hierarchical(path: Path, maxclust: int = None) -> agilkia.TraceSet:
    """
    Load an Agilkia TraceSet and cluster it hierarchically.

    Parameters
    ----------
    path : Path
        Agilkia TraceSet file to load.
    maxclust : Optional[int]
        If given, then the 'maxclust' criterion is used to extact up to this number of cluster.
        If None, then we try to use the same default cutoff as the SciPy dendrogram graphing.

    Returns
    -------
    traces : TraceSet
        Returns the clustered TraceSet.
    """
    traces = agilkia.TraceSet.load_from_json(path)
    print(f"Loaded {len(traces)} traces from {path}")
    data = traces.get_trace_data()  # using default "action_counts" (bag-of-words)

    # standardize the data
    data_scaled = (data - data.mean()) / data.std()
    # for columns with a single value, std() == 0, so we get NaN.  Replace by 0.
    data_scaled.fillna(value=0, inplace=True)

    # %% Use SciPy to create a hierarchical clustering.

    Z = hierarchy.linkage(data_scaled, method='ward')
    print("Top 20 nodes of the tree are [left, right, distance, total_traces]:")
    print(Z[-20:])

    # %% Try to choose the 'best' cut to get flat clusters. (Better done by eye).

    if maxclust is None:
        # we try to do the same as the dendrogram default color algorithm.
        blue = 0.7 * Z[:, 2].max()
        colored = [d for d in Z[:, 2] if d < blue]
        cutoff = max(colored)
        why = f"distance <= {cutoff:.8f}"
        flat = hierarchy.fcluster(Z, cutoff, criterion="distance")
    else:
        why = f"maxclust <= {maxclust}"
        flat = hierarchy.fcluster(Z, maxclust, criterion="maxclust")
    flat = flat - flat.min()  # make it zero-based
    num_clusters = flat.max() + 1
    print(f"Criterion '{why}' chose {num_clusters} clusters.")
    print(sorted(flat))

    # %% Set clusters in TraceSet
    
    traces.set_clusters(flat, linkage=Z)

    # %% Get all the traces in the leftmost tree (root.left)
    
    # root = hierarchy.to_tree(Z)
    # t0 = root.left.pre_order()
    # print(sorted(t0))

    return traces

# %%

def main(args):
    sys.setrecursionlimit(10000)  # to handle larger trees
    if len(args) >= 2:
        maxclust = None
        graph = False
        for name in args[1:]:
            if name == "--graph":
                graph = True
            elif name.startswith("--maxclust="):
                maxclust = int(name.split("=")[1])
            else:
                traces = cluster_hierarchical(Path(name), maxclust)
                if graph:
                    # %% Plot a basic Dendrogram.
                    # plt.figure(figsize=(10, 7))
                    plt.title(f"Dendrogram for {name}")
                    hierarchy.dendrogram(traces.cluster_linkage, p=20,
                                         truncate_mode='level', show_leaf_counts=True)
                    plt.show()
                traces.save_to_json(Path(name).with_suffix(".hier.json"))
    else:
        script = args[0] or "cluster_hierarchical.py"
        print(f"This script reads traces in Agilkia *.json trace format, and")
        print("clusters each TraceSet using the scipy.cluster.hierarchy.linkage() algorithm.")
        print(f"It saves the clustered traces into <input>.hier.json")
        print("")
        print(f"If --maxclust=NUM is given, then up to NUM clusters will be chosen.")
        print(f"Otherwise, the number of clusters will be chosen automatically using a")
        print(f"cophenetic distance heuristic that should match coloring in the dendrogram tree.")
        print("")
        print(f"If '--graph' is specified, then each clustering is graphed as a Dendrogram .")
        print(f"Setup: conda install -c mark.utting agilkia")
        print("")
        print(f"Usage: python {script} [--graph] [--maxclust=NUM] input.json ...")

# %%

if __name__ == "__main__":
    main(sys.argv)

