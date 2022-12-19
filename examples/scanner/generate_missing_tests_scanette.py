#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example analysis of Scanette logs to generate missing tests.

This analysis is described in the paper:

  M. Utting, B. Legeard, F. Dadeau, F. Tamagnan and F. Bouquet,
  "Identifying and Generating Missing Tests using Machine Learning on Execution Traces",
  2020 IEEE International Conference On Artificial Intelligence Testing (AITest),
  Oxford, United Kingdom, 2020, pp. 83-90, doi: 10.1109/AITEST49225.2020.00020.

@author: m.utting@uq.edu.au
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import export_graphviz, export_text
import matplotlib.pyplot as plt
from pathlib import Path
import random
import timeit
import sys
from collections import Counter

import agilkia

# %%

DISPLAY_TIME = 3
TEST_MODE = "--test" in sys.argv
if TEST_MODE:
    print(f"Running in test mode - all figures will close after {DISPLAY_TIME} seconds.")

# %%

INPUT = Path("log_split.json")
cust = agilkia.TraceSet.load_from_json(INPUT)
print(f"Loaded {len(cust)} customer traces from {INPUT}")
print(cust.get_event_chars())
cust.meta_data["dataset"] = "Scanette customer traces"


# %% Get some data about each trace, to use for clustering traces.

data = cust.get_trace_data(method="action_counts")  # or add: method="action_status_counts"
print(data.sum().sort_values())

# %% Now cluster the traces (default MeanShift)

num_clusters = cust.create_clusters(data)
print(num_clusters, "clusters found")
for i in range(num_clusters):
    print(f"Cluster {i}:")
    n = 0
    for tr in cust.get_cluster(i):
        n += 1
        if n > 5:
            print("    ...")
            break
        print(f"    {tr}")


# %% Visualise clusters (using TSNE)

vis = PCA(n_components=2)
cust.visualize_clusters(algorithm=vis, block=(not TEST_MODE))
if TEST_MODE:
    plt.pause(3)
    plt.close()

# %%

print(cust.meta_data)

# %% The event signatures for Scanette

signature = {
    "abandon": {"input":{}, "output":{"Status":"int"}},
    "ajouter": {"input":{}, "output":{"Status":"int"}},
    "debloquer": {"input":{}, "output":{"Status":"int"}},
    "fermerSession": {"input":{}, "output":{"Status":"int"}},
    "ouvrirSession": {"input":{}, "output":{"Status":"int"}},
    "payer": {"input":{}, "output":{"Status":"int"}},
    "scanner": {"input":{}, "output":{"Status":"int"}},
    #"supprimer": {"input":{}, "output":{"Status":"int"}},
    "transmission": {"input":{}, "output":{"Status":"int"}}
    }
    
# %% Some largish clusters that are missing system tests.

# missing = [3, 4, 5]        # use this if you want to evaluate different ML algorithms
missing = []               # use this to skip the ML comparison

# %% Build models for those clusters and evaluate the models.
for n in missing:
    cluster = agilkia.TraceSet(cust.get_cluster(n))
    print(f"========== cluster {n} has {len(cluster)} traces ============")

    # Learn a test-generation model for this cluster.
    ex = agilkia.TracePrefixExtractor()
    X = ex.fit_transform(cluster)
    y = ex.get_labels()
    #print(X.head())
    #print(f"y: {y[0:20]}...")

    # Evaluate various classifiers
    for name,alg in [
                ("Tree", DecisionTreeClassifier()),   # fast, 0.951
                ("GBC", GradientBoostingClassifier()),  # slower, 0.951
                ("RandForest", RandomForestClassifier(n_estimators=100)),  # med 0.951
                ("AdaBoost", AdaBoostClassifier()),  # 0.421, some labels have no predictions
                #("Gaussian", GaussianProcessClassifier(max_iter_predict=10)),   # VERY slow, 0.886
                ("NeuralNet", MLPClassifier(solver='lbfgs')),  # adam solver doesn't converge. 0.924
                ("KNeighbors", KNeighborsClassifier()),  # fast, 0.948
                ("NaiveBayes", GaussianNB()),  # fast, F1 undef. 0.839
                ("LinearSVC", LinearSVC()),  # fast, 0.886
                ("Dummy", DummyClassifier()),  # fast, 0.130
                ("LogReg", LogisticRegression(solver='lbfgs', max_iter=200, multi_class='auto'))  # med 0.89
            ]:
        pipe = Pipeline([
            ("Normalize", MinMaxScaler()),
            (name, alg)
            ])
        scores = cross_val_score(pipe, X, y, cv=10, scoring='f1_macro')
        # print(scores)
        print(f"{name:20s} & {scores.mean():0.3f} (+/- {scores.std() * 2:0.3f})\\\\")


# %% Now use Decision Tree model to generate some tests.
        
def gen_tests_for(traceset, name, traces=5):
    print(f"========== {name} has {len(traceset)} traces ============")

    # Learn a test-generation model for this cluster.
    ex = agilkia.TracePrefixExtractor()
    X = ex.fit_transform(traceset)
    y = ex.get_labels()
    print(f"  it has {len(X)} trace prefixes")
    # Train a decision tree model on this cluster
    model = Pipeline([
        ("Extractor", ex),
        ("Normalize", MinMaxScaler()),
        ("Tree", DecisionTreeClassifier())   # fast, 0.951
        ])
    model.fit(traceset, y)
    
    rand = random.Random(1234)
    smart = agilkia.SmartSequenceGenerator([], method_signatures=signature, rand=rand)
    smart.trace_set.set_event_chars(traceset.get_event_chars())
    # generate some tests
    for i in range(traces):
        smart.generate_trace_with_model(model, length=100)
    for tr in smart.trace_set:
        print(f"    {tr}")
    return (model, smart.trace_set)

# %%
def metric_distinct(traceset: agilkia.TraceSet, n=1) -> float:
    """Calculates the 'distinct-n' metric for all the traces in the given set.

    For example, n=2 means count the number of distinct adjacent pairs of
    action names in the traces, and divide that by the total number of pairs.
    The result will always be between 0.0 to 1.0.
    These metrics are useful as a measure of diversity.
    """
    counts = Counter()
    for tr in traceset:
        for i in range(0, len(tr) - n + 1):
            actions = ",".join([ev.action for ev in tr[i:i+n]])
            counts[actions] += 1
    # print(f"DEBUG: distinct {len(counts)} / {counts}")
    return len(counts) / sum(counts.values())

# %%
for n in [0]:
    cluster = agilkia.TraceSet(cust) # .get_cluster(n))
    (_, tests) = gen_tests_for(cluster, f"cluster {n}")
    print(f"Missing {n} has distinct-1 metric:", metric_distinct(tests))
    print(f"Missing {n} has distinct-2 metric:", metric_distinct(tests, 2))
    print(f"Missing {n} has distinct-3 metric:", metric_distinct(tests, 3))

# %%

(model,_) = gen_tests_for(cust, "all customer traces", traces=30)

# %% Get frequency of a generated trace (maybe partial)

max_length = 35
chars = cust.get_event_chars()


# %%

def freq(trace:agilkia.Trace) -> float:
    return trace.meta_data["freq"]
    
def gen_all():
    smart = agilkia.SmartSequenceGenerator(urls=[], method_signatures=signature, 
                                           verbose=False, action_chars=chars)
    seqs = smart.generate_all_traces(model, length=max_length, action_prob=0.01, path_prob=0.01,
                                     partial=True)
    return seqs

# %% Time the test suite generation
    
duration = timeit.timeit(gen_all, number=10) / 10.0
print(f"duration = {duration} seconds")

# %%

seqs = gen_all()
sorted_seqs = sorted(seqs, key=freq, reverse=True)
print(f"generated {len(seqs)} sequences")
total = 0.0
for tr in sorted_seqs:
    etc = "" if len(tr) < max_length else "???"
    percent = freq(tr) * 100
    total += percent
    print(f"  {percent:5.2f}% {tr.to_string(to_char=chars)}{etc}")
print(f" {total:6.2f}% total")

# %%

ex = model[0]
tree = model[-1]
print(tree)

# %% 

# Export as dot file
export_graphviz(tree, out_file='tree.dot', 
                feature_names = ex.get_feature_names(),
                class_names = model.classes_,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# %% print as text
tree_str = export_text(tree, feature_names=ex.get_feature_names())
print(tree_str)

# %%

if not TEST_MODE:
    # Convert to png using system command (requires Graphviz)
    from subprocess import call
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=300'])

    # Display in jupyter notebook (does not work in Spyder)
    from IPython.display import Image
    Image(filename = 'tree.png')
