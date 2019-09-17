# -*- coding: utf-8 -*-
"""
Custom JSON encoder that can handle arbitrary objects.

Based on ideas from this blog entry by 'The Fellow' (Ouma Rodgers):
https://medium.com/python-pandemonium/json-the-python-way-91aac95d4041

@author: utting@usc.edu.au
"""

import json
import decimal
import unittest


class MyEncoder(json.JSONEncoder):
    """Custom JSON encoder because objects from zeep could not be serialised.
    """
    def default(self, obj):
        if isinstance(obj, (dict, list, tuple, str, int, float, bool)):
            return super().default(obj)  # JSON already handles these
        if isinstance(obj, decimal.Decimal):
            return float(round(obj, 6))  # f"{o:.5f}"
        if isinstance(obj, bytes):
            return "BYTES..."    # or repr(o)
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        if hasattr(obj, "__dict__"):
            result = {
                "__class__": obj.__class__.__name__,
                "__module__": obj.__module__
                }
            result.update(obj.__dict__)
            return result
        raise Exception("JSON serialisation not implemented yet for: " +
                        str(obj) + " type " + str(type(obj)))



class Dummy():
    def __init__(self):
        self.f = [3.14]


class TestMyEncoder(unittest.TestCase):
    """Unit Tests for MyEncoder."""
    def test_decimal(self):
        d1 = decimal.Decimal(3.45)
        d2 = decimal.Decimal(3.4500048012)
        self.assertEqual('3.45', json.dumps(d1, cls=MyEncoder))
        self.assertEqual('3.450005', json.dumps(d2, cls=MyEncoder))

    def test_object(self):
        d1 = Dummy()
        str1 = '{"__class__": "Dummy", "__module__": "__main__", "f": [3.14]}'
        self.assertEqual(str1, json.dumps(d1, cls=MyEncoder))

    def test_nested_object(self):
        d1 = Dummy()
        d2 = Dummy()
        d2.extra = d1
        str1 = '{"__class__": "Dummy", "__module__": "__main__", "f": [3.14]}'
        str2 = str1[0:-1] + ', "extra": ' + str1 + '}'
        self.assertEqual(str2, json.dumps(d2, cls=MyEncoder))

    def test_dict(self):
        d1 = {"a": 1, "b": [2, 3]}
        str1 = '{"a": 1, "b": [2, 3]}'
        self.assertEqual(str1, json.dumps(d1, cls=MyEncoder))

    def test_set(self):
        d = {'testing': {1, 2, 3}}
        str1 = '{"testing": [1, 2, 3]}'
        self.assertEqual(str1, json.dumps(d, cls=MyEncoder))


if __name__ == "__main__":
    unittest.main()
