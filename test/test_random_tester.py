# -*- coding: utf-8 -*-
"""
Unit tests for the RandomTester class.

TODO:
    * test generating more than one trace.
    * test the ML test generator

@author: utting@usc.edu.au
"""

import unittest
import random

import agilkia


WSDL_EG = "http://www.soapclient.com/xml"  # + "/soapresponder.wsdl"

test_input_rules = {
    "username": ["User1"],
    "password": ["<GOOD_PASSWORD>"] * 9 + ["bad-pass"],
    "speed": [str(s) for s in range(0, 120, 10)],
    "bstrParam1": ["VAL1"],
    "bstrParam2": ["p2AAA", "p2BBB"],
}


class TestRandomTester(unittest.TestCase):

    def setUp(self):
        self.tester = agilkia.RandomTester(
                WSDL_EG, ["soapresponder.wsdl"],
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

    def test_dummy_client0(self):
        """Test the dummy web service provided by soapresponder."""
        tester = agilkia.RandomTester(WSDL_EG, ["soapresponder.wsdl"],
                                      input_rules=test_input_rules,
                                      rand=random.Random(1234))
        print("Methods:", tester.get_methods())
        out1 = tester.call_method("Method1")
        self.assertEqual("Your input parameters are VAL1 and p2AAA", out1)
        out1 = tester.call_method("Method1")
        self.assertEqual("Your input parameters are VAL1 and p2AAA", out1)
        out1 = tester.call_method("Method1")
        self.assertEqual("Your input parameters are VAL1 and p2AAA", out1)
        out1 = tester.call_method("Method1")
        self.assertEqual("Your input parameters are VAL1 and p2BBB", out1)
        self.assertEqual(4, len(tester.curr_events))
        self.assertEqual(1, len(tester.trace_set.traces))
        # now generate a second trace
        tester.generate_trace(start=True, length=3)
        self.assertEqual(3, len(tester.curr_events))
        self.assertEqual(2, len(tester.trace_set.traces))
        # now test saving and loading those traces.
        traceset1 = tester.trace_set
        traceset1.save_to_json("tmp_dummy1.json")
        traceset2 = agilkia.TraceSet.load_from_json("tmp_dummy1.json")
        self.assertEqual(traceset2.meta_data, traceset1.meta_data)
        self.assertEqual(len(traceset2.traces), len(traceset1.traces))
        self.assertEqual(len(traceset2.traces[0].events[0]),
                         len(traceset1.traces[0].events[0]))

    def test_generate_trace(self):
        tr = self.tester.generate_trace()
        self.assertEqual(20, len(tr))
