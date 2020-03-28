# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:51:00 2020

@author: Mark Utting
"""

import agilkia
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as hierarchy
import matplotlib.pyplot as plt

# %%

OUTPUT = "traces_0_2_1"

def generate_traceset() -> agilkia.TraceSet:
    ev1 = agilkia.Event("Order", {"Count": 1}, {"Status": 0})
    ev2 = agilkia.Event("Pay", {"Amount": 2}, {"Status": 0})
    ev2b = agilkia.Event("Pay", {"Amount": 3}, {"Status": 1, "Error": "No funds"})

    # a cluster of short traces
    tr0 = agilkia.Trace([ev1])
    tr1 = agilkia.Trace([ev1, ev2])
    tr2 = agilkia.Trace([ev1, ev2b, ev2])
    # a cluster of longer traces
    long0 = agilkia.Trace([ev1, ev1, ev2, ev2] * 4)
    long1 = agilkia.Trace([ev1, ev1, ev2b, ev2] * 4)
    long2 = agilkia.Trace([ev1] * 7)

    traceset = agilkia.TraceSet([tr0, tr1, tr2, long0, long1, long2, tr1])
    traceset.set_event_chars({"Order": "o", "Pay": "P"})
    traceset.set_meta("author", "Mark Utting")
    traceset.set_meta("dataset", "Artificial test traces")
    return traceset

# %%
ts = generate_traceset()
print("Trace lengths:")
print(ts.to_pandas().Trace.value_counts())
print("Trace Data:")
print(ts.get_trace_data())

# %%

# ts.save_to_json(Path(f"fixtures/{OUTPUT}"))

# %% Get bag-of-words data

data = ts.get_trace_data()
print(data)
data_std = (data - data.mean()) / data.std()
print("After standardising:")
print(data_std)

# %% Do a flat clustering using Agilkia (MeanShift)

print("Clusters:", ts.create_clusters(data))
print(ts.cluster_labels)
print(f">>>> SAVED flat clusters into {OUTPUT}_flat.json")
ts.save_to_json(Path(f"fixtures/{OUTPUT}_flat"))

# %% Do a hierarchical clustering using sklearn (for comparison)

model = AgglomerativeClustering()
model.fit(data_std)
print(model)
print(model.labels_)  # [0 0 0 1 1 0 0] two clusters!
print(model.children_)

# %% Do a hierarchical clustering using SciPy linkage.

linkage = hierarchy.linkage(data)
print(linkage)
hierarchy.dendrogram(linkage)
plt.show()

# %% cut the tree to get some flat clusters.

cuts = hierarchy.cut_tree(linkage, n_clusters = [2, 3])
flat2 = cuts[:, 0]
flat3 = cuts[:, 1]
print("cut 2", flat2)  # [0 0 0 1 1 0 0]
print("cut 3", flat3)  # [0 0 0 1 1 2 0]

# %% Save with 2 clusters (just to be different from the flat file) and hierarchy.

print(f">>>> SAVED hierarchical clusters (with 2 flat) into {OUTPUT}_hier.json")
ts.set_clusters(flat2, linkage)
ts.save_to_json(Path(f"fixtures/{OUTPUT}_hier"))


