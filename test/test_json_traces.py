# -*- coding: utf-8 -*-
"""
Test JSON saving and loading.

@author: utting@usc.edu.au
"""

import agilkia
import jsonpickle
import json
import decimal
import datetime
import xml.etree.ElementTree as ET
import os
import unittest
import pytest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class Dummy():
    """Dummy object for testing JSON saving/loading of custom objects."""
    def __init__(self):
        self.f = [3.14]


class TestTraceEncoder(unittest.TestCase):
    """Unit Tests for agilkia.TraceEncoder."""

    xml0 = '{"__class__": "Element", "__module__": "xml.etree.ElementTree", "__tag__": '
    xml1 = xml0 + '"Inner", "__text__": null, "__children__": [], "first": 1, "second": 22}'
    xml2 = xml0 + '"Outer", "__text__": null, "__children__": [' + xml1 + '], "size": 123}'

    def dumps(self, obj):
        """Convenience function that calls json.dumps with appropriate arguments."""
        return json.dumps(obj, cls=agilkia.TraceEncoder)

    def test_decimal(self):
        d1 = decimal.Decimal(3.45)
        d2 = decimal.Decimal(3.4500048012)
        self.assertEqual('3.45', self.dumps(d1))
        self.assertEqual('3.450005', self.dumps(d2))

    def test_object(self):
        d1 = Dummy()
        str1 = '{"__class__": "Dummy", "__module__": "' + __name__ + '", "f": [3.14]}'
        self.assertEqual(str1, self.dumps(d1))

    def test_nested_object(self):
        d1 = Dummy()
        d2 = Dummy()
        d2.extra = d1
        str1 = '{"__class__": "Dummy", "__module__": "' + __name__ + '", "f": [3.14]}'
        str2 = str1[0:-1] + ', "extra": ' + str1 + '}'
        self.assertEqual(str2, self.dumps(d2))

    def test_dict(self):
        d1 = {"a": 1, "b": [2, 3]}
        str1 = '{"a": 1, "b": [2, 3]}'
        self.assertEqual(str1, self.dumps(d1))

    def test_set(self):
        d = {'testing': {1, 2, 3}}
        str1 = '{"testing": [1, 2, 3]}'
        self.assertEqual(str1, self.dumps(d))

    def test_datetime(self):
        d = datetime.datetime(2019, 9, 17, hour=18, minute=58)
        self.assertEqual('"2019-09-17T18:58:00"', self.dumps(d))

    def test_time(self):
        t = datetime.time(hour=18, minute=58)
        self.assertEqual('"18:58:00"', self.dumps(t))

    @pytest.mark.skip(reason="XML objects should be handled separately, by top level.")
    def test_xml1(self):
        xml1 = ET.Element("Inner", attrib={"first": 1, "second": 22})
        self.assertEqual(self.xml1, self.dumps(xml1))
        xml2 = ET.Element("Outer", size=123)
        xml2.append(xml1)
        self.assertEqual(self.xml2, self.dumps(xml2))

    def test_trace(self):
        ev1 = {"action": "Order", "inputs": {"Name": "Mark"}, "outputs": {"Status": 0}}
        tr1 = agilkia.Trace([ev1])
        s0 = "{'__class__': 'Trace', '__module__': 'agilkia.json_traces', 'events': ["
        s1 = str(ev1)
        s2 = "], 'parent': null, 'random_state': null}"

        s0 = '{"__class__": "Trace", "__module__": "agilkia.json_traces", "events": ['
        s1 = '{"action": "Order", "inputs": {"Name": "Mark"}, "outputs": {"Status": 0}}'
        s2 = '], "random_state": null}'
        expect = s0 + s1 + s2
        self.assertEqual(expect, self.dumps(tr1))


class TestXMLDecode(unittest.TestCase):
    """Unit Tests for agilkia.xml_decode."""

    inner = {'first': 1, 'second': 22}

    def test_simple(self):
        xml0 = ET.Element("Simple")
        xml0.text = "abc "
        self.assertEqual("abc ", agilkia.xml_decode(xml0))

    def test_attributes(self):
        xml1 = ET.Element("Inner", attrib=self.inner)
        self.assertEqual(self.inner, agilkia.xml_decode(xml1))

    def test_empty_text(self):
        xml1 = ET.Element("Inner", size=1234)
        xml1.text = "\n    "
        self.assertEqual({'size': 1234}, agilkia.xml_decode(xml1))

    def test_children(self):
        xml = ET.Element("Outer", size=3)
        xml.append(ET.Element("Inner", attrib={"first": 1, "second": 22}))
        vals = ["abc", "def", ""]
        for s in vals:
            x = ET.Element("Child")
            x.text = s
            xml.append(x)
        self.assertEqual({'size': 3, 'Inner': self.inner, 'Child': vals}, agilkia.xml_decode(xml))


class TestJsonTraces(unittest.TestCase):
    """Tests for loading and saving test-case traces."""

    def test_round_trip(self):
        """Test that load and save are the inverse of each other."""
        traces_file = os.path.join(THIS_DIR, "fixtures/traces1.json")
        data2 = agilkia.TraceSet.load_from_json(traces_file)
        self.assertEqual(agilkia.TRACE_SET_VERSION, data2.version)
        data2.save_to_json("tmp2.json")
        data3 = agilkia.TraceSet.load_from_json("tmp2.json")
        self.assertEqual(agilkia.TRACE_SET_VERSION, data3.version)
        self.assertEqual(data2.meta_data, data3.meta_data)
        assert len(data2.traces) == len(data3.traces)
        for i in range(len(data2.traces)):
            # we have not defined equality on Trace objects, so just compare first events.
            self.assertEqual(data3.traces[i].events[0], data2.traces[i].events[0])

    def test_pickled_round_trip(self):
        """Loads some pickled zeep objects and checks that they save/load okay."""
        traces_file = os.path.join(THIS_DIR, "fixtures/traces_pickled.json")
        with open(traces_file, "r") as input:
            data = jsonpickle.loads(input.read())
            print(len(data), "traces loaded")
            parent = agilkia.TraceSet([], {"date": "2019-10-02", "dataset": "test1"})
            for tr in data:
                parent.append(agilkia.Trace(tr))
            parent.save_to_json("tmp.json")
            parent2 = agilkia.TraceSet.load_from_json("tmp.json")
            assert len(data) == len(parent2.traces)

            parent2.save_to_json("tmp2.json")
            parent3 = agilkia.TraceSet.load_from_json("tmp2.json")
            assert len(data) == len(parent3.traces)
            for i in range(len(parent3.traces)):
                self.assertEqual(parent3.traces[i].events[0], parent2.traces[i].events[0])


class TestTrace(unittest.TestCase):
    """Unit tests for agilkia.Trace and agilkia.TraceSet."""

    ev1 = {"action": "Order", "inputs": {"Name": "Mark"}, "outputs": {"Status": 0}}
    ev2 = {"action": "Skip", "inputs": {"Size": 3}, "outputs": {"Status": 1, "Error": "Too big"}}
    ev3 = {"action": "Pay", "inputs": {"Name": "Mark", "Amount": 23.45}, "outputs": {"Status": 0}}
    ev4 = {"action": "Ski", "inputs": {"Type": "downhill"}, "outputs": {"Status": 1}}
    to_char = {"Order": "O", "Skip": ",", "Pay": "p"}

    def test_trace(self):
        tr1 = agilkia.Trace([self.ev2, self.ev1, self.ev3])  # no parent initially
        with self.assertRaises(Exception):
            tr1.to_string()
        self.assertEqual("...", str(tr1))

    def test_traceset(self):
        parent = agilkia.TraceSet([], {})
        # add a first trace
        tr1 = agilkia.Trace([self.ev2, self.ev1, self.ev3])
        parent.append(tr1)
        self.assertEqual(parent, tr1.trace_set())
        self.assertEqual("SOP", tr1.to_string())
        self.assertEqual("SOP", str(tr1))
        # now add a second trace.
        tr2 = agilkia.Trace([self.ev4, self.ev2])
        parent.append(tr2)
        self.assertEqual("Sp", tr2.to_string())
        self.assertEqual("Sp", str(tr2))
        self.assertEqual("pOP", str(tr1))  # changed since to-char is recalculated

    def test_trace_iter(self):
        tr1 = agilkia.Trace([self.ev2, self.ev1, self.ev3])
        it = iter(tr1)
        self.assertEqual(self.ev2, next(it))
        self.assertEqual(self.ev1, next(it))
        self.assertEqual(self.ev3, next(it))
        with self.assertRaises(StopIteration):
            next(it)

    def test_simple(self):
        tr1 = agilkia.Trace([self.ev1, self.ev2, self.ev3])
        s = tr1.to_string(to_char=self.to_char)
        self.assertEqual("O,p", s)

    def test_compress(self):
        events = [self.ev2, self.ev2, self.ev1, self.ev2, self.ev3, self.ev2, self.ev2, self.ev2]
        tr1 = agilkia.Trace(events)
        s = tr1.to_string(to_char=self.to_char)
        self.assertEqual(",,O,p,,,", s)
        s = tr1.to_string(to_char=self.to_char, compress=["Skip"])
        self.assertEqual(",O,p,", s)

    def test_status(self):
        tr = agilkia.Trace([self.ev1, self.ev2, self.ev3])
        s = tr.to_string(to_char=self.to_char, color_status=True)
        self.assertEqual("O\033[91m,\033[0mp", s)

    def test_all_action_names(self):
        tr1 = agilkia.Trace([self.ev1, self.ev3])
        tr2 = agilkia.Trace([self.ev2, self.ev3])
        self.assertEqual(set(["Order", "Skip", "Pay"]), agilkia.all_action_names([tr1, tr2]))

    def test_pandas(self):
        traces = agilkia.TraceSet([])
        traces.append(agilkia.Trace([self.ev1, self.ev3]))
        traces.append(agilkia.Trace([self.ev2, self.ev3]))
        df = traces.to_pandas()
        self.assertEqual(4, df.shape[0])  # rows
        self.assertEqual(8, df.shape[1])  # columns
        cols = ["trace", "event", "action", "Status", "Error", "Name", "Amount", "Size"]
        self.assertEqual(cols, list(df.columns))

    def test_default_meta_data(self):
        now = str(datetime.datetime.now())
        md = agilkia.TraceSet.get_default_meta_data()
        self.assertEqual("pytest", md["source"].split("/")[-1])
        self.assertEqual(now[0:10], md["date"][0:10])  # same date, unless it is exactly midnight!


class TestDefaultMapToChars(unittest.TestCase):

    def test_default_map_to_chars(self):
        actions = ["Order", "Skip", "PayLate", "Pay"]
        expect = {"Order": "O", "Skip": "S", "PayLate": "L", "Pay": "P"}
        self.assertEqual(expect, agilkia.default_map_to_chars(actions))

    def test_default_map_to_chars_prefixes(self):
        actions = ["Order", "PayLate", "PayEarly", "PayExtra"]
        expect = {'Order': 'O', 'PayEarly': 'a', 'PayExtra': 'x', 'PayLate': 'L'}
        self.assertEqual(expect, agilkia.default_map_to_chars(actions))

    def test_default_map_to_chars_hard(self):
        actions = ["O", "Oa", "Oy", "Pay", "Play", "yay"]
        expect = {'O': 'O', 'Oa': 'a', 'Oy': 'y', 'Pay': 'P', 'Play': 'l', 'yay': '*'}
        self.assertEqual(expect, agilkia.default_map_to_chars(actions))

    def test_default_map_to_chars_given(self):
        actions = ["Order", "Save", "Skip", "PayLate", "Pay"]
        given = {"Save": "."}
        expect = {'Order': 'O', 'Pay': 'P', 'PayLate': 'L', 'Save': '.', 'Skip': 'S'}
        self.assertEqual(expect, agilkia.default_map_to_chars(actions, given=given))


if __name__ == "__main__":
    unittest.main()
