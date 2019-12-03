# -*- coding: utf-8 -*-
"""
Unit tests for the RandomTester class.

TODO:
    * test the ML test generator

@author: utting@usc.edu.au
"""

import unittest
import random
from pathlib import Path
import sklearn.utils.estimator_checks

import agilkia

THIS_DIR = Path(__file__).parent
WSDL_EG = "http://www.soapclient.com/xml/soapresponder.wsdl"

test_input_rules = {
    "username": ["User1"],
    "password": ["<GOOD_PASSWORD>"] * 9 + ["bad-pass"],
    "speed": [str(s) for s in range(0, 120, 10)],
    "bstrParam1": ["VAL1"],
    "bstrParam2": ["p2AAA", "p2BBB"],
}


class TestReadInputRules(unittest.TestCase):

    def test_1(self):
        rules = agilkia.read_input_rules(THIS_DIR / "fixtures/inputs1.csv")
        self.assertEqual(["one"], rules["bstrParam1"])
        self.assertEqual(['two', 'two', 'two', 'TWO!'], rules["bstrParam2"])


class TestRandomTester(unittest.TestCase):

    def setUp(self):
        self.tester = agilkia.RandomTester(
                WSDL_EG,
                input_rules=test_input_rules,
                rand=random.Random(1234))

    def test_input_user(self):
        self.assertEqual("User1", self.tester.choose_input_value("username"))

    def test_input_password(self):
        self.assertEqual(agilkia.GOOD_PASSWORD, self.tester.choose_input_value("password"))

    def test_input_speeds(self):
        speeds = set()
        for i in range(100):
            speeds.add(self.tester.choose_input_value("speed"))
        self.assertEqual(12, len(speeds))  # all results should be covered

    def test_signature(self):
        sig = self.tester.get_methods()
        self.assertEqual(1, len(sig))
        self.assertEqual({"Method1"}, sig.keys())
        msig = sig["Method1"]
        self.assertEqual(1, len(msig))
        self.assertEqual({"input"}, msig.keys())
        self.assertEqual({"bstrParam1", "bstrParam2"}, msig["input"].keys())
        param1_details = "{'optional': False, 'type': 'String(value)'}"
        self.assertEqual(param1_details, str(msig["input"]["bstrParam1"]))

    def test_dummy_client_meta(self):
        """Test the dummy web service provided by soapresponder."""
        tester = agilkia.RandomTester(WSDL_EG,
                                      input_rules=test_input_rules,
                                      rand=random.Random(1234))
        meta_keys = ["date", "author", "dataset", "source",
                     "web_services", "methods_to_test", "input_rules",
                     "method_signatures"]
        mdata = tester.trace_set.meta_data
        for k in meta_keys:
            self.assertTrue(k in mdata, msg=k + " expected in meta_data")
        self.assertEqual(f"RandomTester", mdata["source"])
        self.assertEqual([WSDL_EG], mdata["web_services"])
        # check the signature
        self.assertEqual(set(["Method1"]), set(mdata["method_signatures"].keys()))
        sig = {'input': {
               'bstrParam1': {'optional': False, 'type': 'String(value)'},
               'bstrParam2': {'optional': False, 'type': 'String(value)'}}}
        self.assertEqual(sig, mdata["method_signatures"]["Method1"])

    def test_dummy_client0(self):
        """Test the dummy web service provided by soapresponder."""
        tester = agilkia.RandomTester(WSDL_EG,
                                      input_rules=test_input_rules,
                                      rand=random.Random(1234))
        print("Methods:", tester.get_methods())
        out1 = tester.call_method("Method1")
        expect = {"Status": 0, "string": "Your input parameters are VAL1 and p2AAA"}
        self.assertEqual(expect, out1.outputs)
        out1 = tester.call_method("Method1")
        self.assertEqual(expect, out1.outputs)
        out1 = tester.call_method("Method1")
        self.assertEqual(expect, out1.outputs)
        out1 = tester.call_method("Method1")
        expect["string"] = "Your input parameters are VAL1 and p2BBB"
        self.assertEqual(expect, out1.outputs)
        self.assertEqual(4, len(tester.curr_events))
        self.assertEqual(1, len(tester.trace_set.traces))
        # now generate a second trace
        tester.generate_trace(start=True, length=3)
        self.assertEqual(3, len(tester.curr_events))
        self.assertEqual(2, len(tester.trace_set.traces))
        # now test saving and loading those traces.
        traceset1 = tester.trace_set
        tmp_json = Path("tmp_dummy1.json")
        traceset1.save_to_json(tmp_json)
        traceset2 = agilkia.TraceSet.load_from_json(tmp_json)
        self.assertEqual(traceset2.meta_data, traceset1.meta_data)
        self.assertEqual(len(traceset2.traces), len(traceset1.traces))
        self.assertEqual(traceset2.traces[0].events[0].action,
                         traceset1.traces[0].events[0].action)
        tmp_json.unlink()

    def test_generate_trace(self):
        tr = self.tester.generate_trace()
        self.assertTrue(isinstance(tr, agilkia.Trace))
        self.assertEqual(20, len(tr.events))

    def test_decode_outputs(self):
        self.assertEqual({'Status': 3}, self.tester.decode_outputs(3))
        self.assertEqual({'Status': 0, "string": "abc"}, self.tester.decode_outputs("abc"))
        self.assertEqual({'Status': 0, "a": 2}, self.tester.decode_outputs({"a": 2}))
        # Also, zeep XML object outputs are tested in test_dummy_client0 above.


class TestTracePrefixExtractor(unittest.TestCase):

    ev1 = agilkia.Event("Order", {"Name": "Mark"}, {"Status": 0})
    ev2 = agilkia.Event("Skip", {"Size": 3}, {"Status": 1, "Error": "Too big"})
    ev3 = agilkia.Event("Pay", {"Name": "Mark", "Amount": 23.45}, {"Status": 0})
    
    def test_bag_of_words(self):
        tr1 = agilkia.Trace([self.ev1, self.ev2])
        tr2 = agilkia.Trace([self.ev3])
        traces = agilkia.TraceSet([tr1, tr1, tr2])
        self.assertEquals(3, len(traces))
        sut = agilkia.TracePrefixExtractor()
        sut.fit(traces)
        self.assertEqual(["Order", "Pay", "Skip"], sut.get_feature_names())
        X = sut.transform(traces)
        y = sut.get_labels()
        self.assertEqual((8, 3), X.shape)
        self.assertEqual(8, len(y))
        for row in [0, 3, 6]:
            self.assertEqual([0.0, 0.0, 0.0], X.iloc[row, :].tolist())
            self.assertEqual("Order" if row < 6 else "Pay", y[row])
        for row in [2, 5]:
            self.assertEqual([1.0, 0.0, 1.0], X.iloc[row, :].tolist())
            self.assertEqual(agilkia.TRACE_END, y[row])
        self.assertEqual([0.0, 1.0, 0.0], X.iloc[7, :].tolist())
        
        