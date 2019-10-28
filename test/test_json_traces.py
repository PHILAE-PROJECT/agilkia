# -*- coding: utf-8 -*-
"""
Test JSON saving and loading.

@author: utting@usc.edu.au
"""

import agilkia
import jsonpickle     # type: ignore
import json
import decimal
import datetime
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd   # type: ignore
import unittest
import pytest         # type: ignore

THIS_DIR = Path(__file__).parent


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
        ev1 = agilkia.Event("Order", {"Name": "Mark"}, {"Status": 0})
        tr1 = agilkia.Trace([ev1])
        s0 = '{"__class__": "Trace", "__module__": "agilkia.json_traces", "events": ['
        s1a = '{"__class__": "Event", "__module__": "agilkia.json_traces", "action": "Order", '
        s1b = '"inputs": {"Name": "Mark"}, "outputs": {"Status": 0}, "meta_data": {}}'
        s2 = '], "meta_data": {}}'
        expect = s0 + s1a + s1b + s2
        self.maxDiff = None
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
    """Tests for loading and saving Event, Trace and TraceSet objects."""

    def test_round_trip(self):
        """Test that load and save are the inverse of each other."""
        data2 = agilkia.TraceSet.load_from_json(THIS_DIR / "fixtures/traces1.json")
        self.assertEqual(agilkia.TRACE_SET_VERSION, data2.version)
        tmp2_json = Path("tmp2.json")
        data2.save_to_json(tmp2_json)
        data3 = agilkia.TraceSet.load_from_json(tmp2_json)
        self.assertEqual(agilkia.TRACE_SET_VERSION, data3.version)
        self.assertEqual(data2.meta_data, data3.meta_data)
        assert len(data2.traces) == len(data3.traces)
        for i in range(len(data2.traces)):
            # we have not defined equality on Trace objects, so just compare first events.
            ev3 = data3.traces[i].events[0]
            ev2 = data2.traces[i].events[0]
            self.assertEqual(ev3.action, ev2.action)
            self.assertEqual(ev3.inputs, ev2.inputs)
            self.assertEqual(ev3.status, ev2.status)
        # if all went well, we can delete the temp file now.
        tmp2_json.unlink()

    def test_pickled_round_trip(self):
        """Loads some pickled zeep objects and checks that they save/load okay."""
        pickled = THIS_DIR / "fixtures/traces_pickled.json"
        # with open(traces_file, "r") as input:
        data = jsonpickle.loads(pickled.read_text())
        print(len(data), "traces loaded")
        parent = agilkia.TraceSet([], {"date": "2019-10-02", "dataset": "test1"})
        for events in data:
            trace = agilkia.Trace([])
            for e in events:
                event = agilkia.Event(e["action"], e["inputs"], e["outputs"])
                trace.append(event)
            parent.append(trace)
        tmp_json = Path("tmp.json")
        tmp2_json = Path("tmp2.json")
        parent.save_to_json(tmp_json)
        parent2 = agilkia.TraceSet.load_from_json(tmp_json)
        assert len(data) == len(parent2.traces)

        parent2.save_to_json(tmp2_json)
        parent3 = agilkia.TraceSet.load_from_json(tmp2_json)
        assert len(data) == len(parent3.traces)
        for i in range(len(parent3.traces)):
            ev3 = parent3.traces[i].events[0]
            ev2 = parent2.traces[i].events[0]
            self.assertEqual(ev3.action, ev2.action)
            self.assertEqual(ev3.inputs, ev2.inputs)
        # if all went well, we can delete the temp files now.
        tmp_json.unlink()
        tmp2_json.unlink()


class TestTrace(unittest.TestCase):
    """Unit tests for agilkia.Trace and agilkia.TraceSet."""

    ev1 = agilkia.Event("Order", {"Name": "Mark"}, {"Status": 0})
    ev2 = agilkia.Event("Skip", {"Size": 3}, {"Status": 1, "Error": "Too big"})
    ev3 = agilkia.Event("Pay", {"Name": "Mark", "Amount": 23.45}, {"Status": 0})
    ev4 = agilkia.Event("Ski", {"Type": "downhill"}, {"Status": 1})
    to_char = {"Order": "O", "Skip": ",", "Pay": "p"}

    def test_trace(self):
        tr1 = agilkia.Trace([self.ev2, self.ev1, self.ev3])  # no parent initially
        with self.assertRaises(Exception):
            tr1.to_string()
        self.assertEqual("...", str(tr1))
        self.assertEqual(3, len(tr1))
        self.assertEqual(self.ev2, tr1[0])
        self.assertEqual(self.ev3, tr1[-1])

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
        self.assertEqual(2, len(parent))
        self.assertEqual(tr1, parent[0])
        self.assertEqual(tr2, parent[-1])

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
        cols = ["Trace", "Event", "Action", "Status", "Error", "Name", "Amount", "Size"]
        self.assertEqual(cols, list(df.columns))

    def test_default_meta_data(self):
        now = str(datetime.datetime.now())
        md = agilkia.TraceSet.get_default_meta_data()
        self.assertEqual("pytest", md["source"].split("/")[-1])
        self.assertEqual(now[0:10], md["date"][0:10])  # same date, unless it is exactly midnight!

    def test_arff_type(self):
        traces = agilkia.TraceSet([])
        i64 = pd.api.types.pandas_dtype("int64")
        self.assertEqual("INTEGER", traces.arff_type(i64))

    def test_split_none(self):
        # each ev1 will start a new trace
        trace = agilkia.Trace([self.ev2, self.ev1, self.ev3, self.ev1, self.ev1, self.ev2])
        traces = agilkia.TraceSet([trace])
        with self.assertRaises(Exception):
            traces.with_traces_split()
        traces2 = traces.with_traces_split(start_action="XYZ")
        self.assertEqual(1, len(traces2))  # none split

    def test_split_action(self):
        # each ev1 will start a new trace
        trace = agilkia.Trace([self.ev2, self.ev1, self.ev3, self.ev1, self.ev1, self.ev2])
        traces = agilkia.TraceSet([trace])
        traces2 = traces.with_traces_split(start_action="Order")
        self.assertEqual(4, len(traces2))
        self.assertEqual("Skip", traces2[0].events[0].action)
        self.assertEqual("Order", traces2[1].events[0].action)
        self.assertEqual("Order", traces2[2].events[0].action)
        self.assertEqual("Order", traces2[3].events[0].action)

    def test_split_input(self):
        ev3b = agilkia.Event("Pay", {"Name": "Merry", "Amount": 23.45}, {"Status": 0})
        # each change in the "Name" input will start a new trace
        trace = agilkia.Trace([self.ev1, self.ev3, ev3b, self.ev3, self.ev2, self.ev1])
        traces = agilkia.TraceSet([trace])
        traces2 = traces.with_traces_split(input_name="Name")
        for t in traces2:
            print(t)
        self.assertEqual(3, len(traces2))
        self.assertEqual("Mark", traces2[0].events[0].inputs["Name"])
        self.assertEqual("Merry", traces2[1].events[0].inputs["Name"])
        self.assertEqual("Mark", traces2[2].events[0].inputs["Name"])
        self.assertEqual(2, len(traces2[0]))
        self.assertEqual(1, len(traces2[1]))
        self.assertEqual(3, len(traces2[2]))

    def test_group_input(self):
        ev3b = agilkia.Event("Pay", {"Name": "Merry", "Amount": 23.45}, {"Status": 0})
        # each different "Name" input will be grouped into a new trace
        trace = agilkia.Trace([self.ev1, self.ev3, ev3b, self.ev3, self.ev2, self.ev1])
        traces = agilkia.TraceSet([trace])
        traces2 = traces.with_traces_grouped_by("Name")  # self.ev2 will be discarded
        self.assertEqual(2, len(traces2))
        self.assertEqual("Mark", traces2[0].events[0].inputs["Name"])
        self.assertEqual("Merry", traces2[1].events[0].inputs["Name"])
        self.assertEqual(4, len(traces2[0]))
        self.assertEqual(1, len(traces2[1]))


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


class TestSafeString(unittest.TestCase):

    def test_safe_string(self):
        self.assertEqual("Ab9", agilkia.safe_name("Ab9"))
        self.assertEqual("_Ab9_etc_json", agilkia.safe_name("!Ab9 etc.json"))


if __name__ == "__main__":
    unittest.main()
