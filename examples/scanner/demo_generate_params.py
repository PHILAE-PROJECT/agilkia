"""A simple demo of generating traces from ML models, including models for parameter values.

The generation of abstract sequences of actions is the same as in the AITest 2020 paper::

    * Utting, Mark, Legeard, Bruno, Dadeau, Frederic, Tamagnan, Frederic and Bouquet, Fabrice (2020). 
    * Identifying and generating missing tests using machine learning on execution traces.
    * IEEE International Conference on Artificial Intelligence Testing (AITest),
    * Oxford, United Kingdom, 3-6 August 2020.

The new focus here is the generation of parameter values, using various random / ML approaches.

Author: Mark Utting, Dec 2021.
"""

from agilkia import TraceSet, TracePrefixExtractor, SmartSequenceGenerator
from agilkia import SessionGenerator, CategoricalGenerator, NumericalGenerator
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


METHODS = {
    "abandon": {"input":{}, "output":{"Status":"int"}},
    "ajouter": {"input":{}, "output":{"Status":"int"}},
    "debloquer": {"input":{}, "output":{"Status":"int"}},
    "fermerSession": {"input":{}, "output":{"Status":"int"}},
    "ouvrirSession": {"input":{}, "output":{"Status":"int"}},
    "payer": {"input":{}, "output":{"Status":"int"}},
    "scanner": {"input":{}, "output":{"Status":"int"}},
    "supprimer": {"input":{}, "output":{"Status":"int"}},
    "transmission": {"input":{}, "output":{"Status":"int"}}
    }


training = TraceSet.load_from_json(Path("100043-steps.split.json"))

# learn a test-generation model of action sequences
# -------------------------------------------------
ex = TracePrefixExtractor() # (event_to_str=lambda ev: f"{ev.action}_{ev.status}")
X = ex.fit_transform(training)
y = ex.get_labels()
print(f"training data has {len(training)} traces, {len(X)} prefixes")
# Train a decision tree model on this cluster
model = Pipeline([
    ("Extractor", ex),
    ("Normalize", MinMaxScaler()),
    ("Tree", DecisionTreeClassifier())
    ])
model.fit(training, y)


# generate a few test traces from the model
# -----------------------------------------
smart = SmartSequenceGenerator([], method_signatures=METHODS)
smart.trace_set.set_event_chars(training.get_event_chars())
for i in range(10):
    events = smart.generate_trace_with_model(model, length=30)
generated = smart.trace_set
for i,tr in enumerate(generated):
    print(f"  generated {i}:  {tr}")


# generate all missing input or output values
# -------------------------------------------
fields = {
    "Action": "categorical", 
    "Status": "numerical",
    "sessionID": "categorical",
    "object": "categorical",
    "param": "categorical"
    }
generators = [SessionGenerator(fields, current_index=2, prefix="client"),
              CategoricalGenerator(fields, current_index=3),
              CategoricalGenerator(fields, current_index=4)]

print("Trace 0, showing missing inputs:")
for ev in generated[0]:
    print(ev)

for i,g in enumerate(generators):
    print("  learning column:", i + 1)
    g.fit(training)
    print("  generating column:", i + 1)
    g.transform(generated)

print("Trace 0, showing generated inputs:")
for ev in generated[0]:
    print(ev)

generated.save_to_json(Path("10-generated-traces.agilkia.json"))

