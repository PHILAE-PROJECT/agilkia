# Agilkia: A Python Toolkit to Support AI-for-Testing

This toolkit is intended to make it easier to build testing tools that learn
from traces of customer behaviors, analyze those traces for common patterns
and unusual behaviors (e.g. using clustering techniques), learn machine learning (ML)
models of typical behaviors, and use those models to generate smart tests that
imitate customer behaviors.

Agilkia is intended to provide a storage and interchange format that makes it easy to
built 'smart' tools on top of this toolkit, often with just a few lines of code.
The main focus of this toolkit is saving and loading traces in a standard *.JSON
format, and transforming those traces to and from lots of other useful formats,
including:

 * Pandas DataFrames (for data analysis and machine learning);
 * ARFF files (for connection to Weka and the StackedTrees tools);
 * SciPy Linkage matrices (for hierarchical clustering and drawing Dendrograms);
 * CSV files in application-specific formats (requires writing some Python code).

The key datastructures supported by this library are:

 * TraceSet = a sequence of Trace objects
 * Trace = a sequence of Event objects
 * Event = one interaction with a web service/site, with an action name, inputs, outputs.

In addition, note that the TraceSet can store 'clustering' information about the
traces (flat clusters and optional hierarchical clustering) and all three of the
above objects include various kinds of 'meta-data'.  For example, each Event
object can contain a timestamp, and each TraceSet contains an 'event_chars' dictionary
that maps each kind of event to a single character to enable concise visualization of traces.


This 'agilkia' library is part of the Philae research project:

    http://projects.femto-st.fr/philae/en

It is open source software under the MIT license.
See LICENSE.txt

# Key Features:

* Manage sets of traces (load/save to JSON, etc.).
* Split traces into smaller traces (sessions).
* Cluster traces on various criteria, with support for flat and hierarchical clustering.
* Visualise clusters of tests, to see common / rare behaviours.
* Convert traces to Pandas DataFrame for data analysis / machine learning.
* Generate random tests, or 'smart' tests from a machine learning (ML) model.
* Automated testing of SOAP web services with WSDL descriptions.


## About the Name

The name 'Agilkia' was chosen for this library because it is
closely associated with the name 'Philae', and the Agilkia toolkit
has been developed as part of the Philae research project.

Agilkia is an island in the reservoir of the Aswan Low Dam, 
downstream of the Aswan Dam and Lake Nasser, Egypt.  
It is the current location of the ancient temple of Isis, which was 
moved there from the islands of Philae after dam water levels rose.
    
Agilkia was also the name given to the first landing place of the
Philae landing craft on the comet 67P/Churyumov–Gerasimenko,
during the Rosetta space mission.


## People

* Architect and developer: AProf. Mark Utting
* Project leader: Prof. Bruno Legeard


# Example Usages

Agilkia requires Python 3.7 or higher.
Here is how to install this toolkit using conda:
```
conda install -c mark.utting agilkia
```

Here is a simple example of generating some simple random tests of an imaginary
web service running on the URL http://localhost/cash:
```
import agilkia

# sample input values for named parameters
input_values = {
    "username"  : ["TestUser"],
    "password"  : ["<GOOD_PASSWORD>"] * 9 + ["bad-pass"],  # wrong 10% of time
    "version"   : ["2.7"] * 9 + ["2.6"],      # old version 10% of time
    "account"   : ["acc100", "acc103"],       # test these two accounts equally
    "deposit"   : [i*100 for i in range(8)],  # a range of deposit amounts
}

def first_tests():
    tester = agilkia.RandomTester("http://localhost/cash",
        parameters=input_values)
    tester.set_username("TestUser")   # will prompt for password
    tests = agilkia.TraceSet([])
    for i in range(10):
        tr = tester.generate_trace(length=30)
        print(f"========== trace {i}:\n  {tr}")
        tests.append(tr)
    return tests

first_tests().save_to_json(Path("tests1.json"))
```

And here is an example of loading a file containing a single long trace, splitting it into
customer sessions based on a 'sessionID' input field, using SciPy to cluster those sessions
using hierarchical clustering, visualizing them as a dendrogram tree, and saving the results.
```
from pathlib import Path
import scipy.cluster.hierarchy as hier
import matplotlib.pyplot as plt
import agilkia

traces = agilkia.TraceSet.load_from_json(Path("trace.json"))
sessions = traces.with_traces_grouped_by("sessionID")
data = sessions.get_trace_data(method="action_counts")
tree = hier.linkage(data)
hier.dendrogram(tree, 10, truncate_mode="level")  # view top 10 levels of tree
plt.show()
cuts = hier.cut_tree(tree, [3])                   # cut the tree to get 3 clusters
sessions.set_clusters(cuts[:,0], tree)
sessions.save_to_json(Path("sessions_clustered.json"))
```

For more complete examples, see the *.py scripts in the `examples/scanner` directory in the
Agilkia source code distribution (`https://github.com/utting/agilkia`) and the README there.
