#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example analysis of Scanette logs to generate missing tests.

@author: utting@usc.edu.au
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

import agilkia

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
cust.visualize_clusters(algorithm=vis)

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
    
# %% Generate smart tests for several missing clusters.

missing = [3, 4, 5]
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
        print(f"{name:20s} & {scores.mean():0.3f} (+/- {scores.std() * 2:0.3f})")


# %% Now use Decision Tree to generate some tests.
for n in missing:
    cluster = agilkia.TraceSet(cust.get_cluster(n))
    print(f"========== cluster {n} has {len(cluster)} traces ============")

    # Learn a test-generation model for this cluster.
    ex = agilkia.TracePrefixExtractor()
    X = ex.fit_transform(cluster)
    y = ex.get_labels()
    print(f"len(y) = {len(y)}")
    # Train a decision tree model on this cluster
    model = Pipeline([
        ("Extractor", ex),
        ("Normalize", MinMaxScaler()),
        ("Tree", DecisionTreeClassifier())   # fast, 0.951
        ])
    model.fit(cluster, y)
    
    rand = random.Random(1234)
    smart = agilkia.SmartSequenceGenerator(methods=signature, verbose=False, rand=rand)
    smart.trace_set.set_event_chars(cluster.get_event_chars())
    # generate some tests
    for i in range(5):
        smart.generate_trace_with_model(model, length=100)
    for tr in smart.trace_set:
        print(f"    {tr}")

# %% Experiment with generating ALL likely sequences.

# Learn a test-generation model for all the customer traces.
ex = agilkia.TracePrefixExtractor()
X = ex.fit_transform(cust)
y = ex.get_labels()
# Train a decision tree model on this cluster
tree = DecisionTreeClassifier()
model = Pipeline([
    ("Extractor", ex),
    ("Normalize", MinMaxScaler()),
    ("Tree", tree)
    ])
model.fit(cust, y)
print(tree)

# %%

chars = cust.get_event_chars()
smart = agilkia.SmartSequenceGenerator(methods=signature, verbose=False,
                                       action_chars=chars)
seqs = smart.generate_all_traces(model, length=14, action_prob=0.01, path_prob=1e-2,
                                 partial=False)
print(f"generated {len(seqs)} sequences")
for tr in seqs:
    print(f"    {tr.to_string(to_char=chars)}")

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

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# %%
# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')
