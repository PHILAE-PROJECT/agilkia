# -*- coding: utf-8 -*-
"""
Unit tests for the SmartSequenceGenerator class.

@author: m.utting@uq.edu.au
"""

import unittest
import random
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

import agilkia

THIS_DIR = Path(__file__).parent

test_input_rules = {
    "username": ["User1"],
    "password": ["<GOOD_PASSWORD>"] * 9 + ["bad-pass"],
    "speed": [str(s) for s in range(0, 120, 10)],
    "bstrParam1": ["VAL1"],
    "bstrParam2": ["p2AAA", "p2BBB"],
}

# %% The event signatures for a dummy website that sells products.

signature = {
    "start": {"input": {}, "output": {"Status": "int"}},
    "browse": {"input": {}, "output": {"Status": "int"}},
    "add": {"input": {}, "output": {"Status": "int"}},
    "pay": {"input": {}, "output": {"Status": "int"}}
    }

event_start0 = agilkia.Event("start", {}, {"Status": 0})
event_browse0 = agilkia.Event("browse", {}, {"Status": 0})
event_add0 = agilkia.Event("add", {}, {"Status": 0})
event_pay0 = agilkia.Event("pay", {}, {"Status": 0})


def gen_dummy_traces(rand: random.Random, n=10) -> agilkia.TraceSet:
    """Generate a dummy set of n traces: Start;(Browse;Add)*;Pay."""
    traces = agilkia.TraceSet([])
    for i in range(n):
        tr = agilkia.Trace([])
        traces.append(tr)
        tr.append(event_start0)
        for j in range(rand.randint(1, 4)):
            tr.append(event_browse0)
            tr.append(event_add0)
        tr.append(event_pay0)
        # print(f"{i}: {tr}")
    return traces


class TestSmartSequenceGenerator(unittest.TestCase):

    def setUp(self):
        self.tester = agilkia.SmartSequenceGenerator(
                [],
                method_signatures=signature,
                input_rules=test_input_rules,
                rand=random.Random(1234)
        )
        self.rand = random.Random(12345)
        self.training = gen_dummy_traces(self.rand, 100)
        # Learn a test-generation model for this website.
        ex = agilkia.TracePrefixExtractor()
        # we must calculate all trace prefixes, just to know the output labels.
        ex.fit_transform(self.training)
        y = ex.get_labels()
        # Train a decision tree model on this cluster
        self.model = Pipeline([
            ("Extractor", ex),
            ("Normalize", MinMaxScaler()),
            ("Tree", DecisionTreeClassifier())
        ])
        self.model.fit(self.training, y)

    def test_generate_trace_with_model(self):
        smart = agilkia.SmartSequenceGenerator([], method_signatures=signature, rand=self.rand)
        # generate some tests
        gen = agilkia.TraceSet([])
        for i in range(5):
            evs = smart.generate_trace_with_model(self.model, length=100)
            tr = agilkia.Trace(evs)
            gen.append(tr)
            # print(f"  generated {i}: {tr}")
        self.assertEqual("sbap", str(gen[0]))
        self.assertEqual("sbababap", str(gen[1]))
        self.assertEqual("sbabababap", str(gen[2]))
        self.assertEqual("sbabababap", str(gen[3]))
        self.assertEqual("sbabap", str(gen[4]))

    def test_generate_all_traces(self):
        """Test the generation of all traces with probability at least 1%."""
        smart = agilkia.SmartSequenceGenerator([], method_signatures=signature, rand=self.rand)
        traces = smart.generate_all_traces(self.model, length=7, partial=True)
        traceset = agilkia.TraceSet(traces)
        expect = [
            ("sbababa", 0.40),  # cut short before final 'p', due to length=7
            ("sbabap", 0.32),
            ("sbap", 0.28)
        ]
        self.assertEqual(3, len(traceset))
        for i in range(len(expect)):
            (s, f) = expect[i]
            tr = traceset[i]
            freq = tr.get_meta("freq")
            # print(f"  {tr}  {freq}")
            self.assertEqual(s, str(tr))
            self.assertAlmostEqual(f, freq)
