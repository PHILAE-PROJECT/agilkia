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
    """Unit Tests for MyEncoder."""

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


class TestXMLDecode(unittest.TestCase):
    """Unit Tests for agilkia.xml_decode."""

    inner = {'first': 1, 'second': 22}

    def test_simple(self):
        xml0 = ET.Element("Simple")
        xml0.text = "abc"
        self.assertEqual("abc", agilkia.xml_decode(xml0))

    def test_attributes(self):
        xml1 = ET.Element("Inner", attrib=self.inner)
        self.assertEqual(self.inner, agilkia.xml_decode(xml1))

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
        data2 = agilkia.load_traces_from_json(traces_file)
        agilkia.save_traces_to_json(data2, "tmp2.json")
        data3 = agilkia.load_traces_from_json("tmp2.json")
        assert len(data2) == len(data3)
        for i in range(len(data2)):
            self.assertEqual(data3[i], data2[i])

    def test_pickled_round_trip(self):
        """Loads some pickled zeep objects and checks that they save/load okay."""
        traces_file = os.path.join(THIS_DIR, "fixtures/traces_pickled.json")
        with open(traces_file, "r") as input:
            data = jsonpickle.loads(input.read())
            print(len(data), "traces loaded")
            agilkia.save_traces_to_json(data, "tmp.json")
            data2 = agilkia.load_traces_from_json("tmp.json")
            assert len(data) == len(data2)

            agilkia.save_traces_to_json(data, "tmp2.json")
            data3 = agilkia.load_traces_from_json("tmp2.json")
            assert len(data) == len(data3)
            for i in range(len(data2)):
                self.assertEqual(data3[i], data2[i])


if __name__ == "__main__":
    unittest.main()
