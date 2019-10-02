# -*- coding: utf-8 -*-
"""
Unit tests for the RandomTester class.

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
        self.assertEqual(4, len(tester.curr_trace))
        self.assertEqual(1, len(tester.all_traces))
        # TODO:
        # agilkia.save_traces_to_json(tester.all_traces, "tmp_dummy1.json")
        # traces2 = agilkia.load_traces_from_json("tmp_dummy1.json")
        # self.assertEqual(traces2, tester.all_traces)

    def test_generate_trace(self):
        tr = self.tester.generate_trace()
        self.assertEqual(20, len(tr))
