# -*- coding: utf-8 -*-
"""
Custom JSON encoder that can handle arbitrary objects.

Based on ideas from this blog entry by 'The Fellow' (Ouma Rodgers):
https://medium.com/python-pandemonium/json-the-python-way-91aac95d4041

@author: utting@usc.edu.au
"""

import json
import decimal
import datetime
from typing import List


class TraceEncoder(json.JSONEncoder):
    """Custom JSON encoder because objects from zeep could not be serialised."""

    # count = 0

    def default(self, obj):
        if isinstance(obj, (dict, list, tuple, str, int, float, bool)):
            return super().default(obj)  # JSON already handles these
        if isinstance(obj, decimal.Decimal):
            return float(round(obj, 6))  # f"{o:.5f}"
        if isinstance(obj, (bytes, bytearray)):
            return "BYTES..."    # TODO: handle these better: repr(o)?
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        if isinstance(obj, (datetime.date, datetime.datetime, datetime.time)):
            return obj.isoformat()  # as a string
        if hasattr(obj, "__dict__"):
            result = {
                "__class__": obj.__class__.__name__,
                "__module__": obj.__module__
                }
#            if self.count < 10:
#                print(obj.__class__.__name__, obj.__dict__)
#                self.count += 1
            if len(obj.__dict__) == 1 and "__values__" in obj.__dict__:
                # zeep seems to hide the attributes in a __values__ dict.
                # We lift them up to the top level to make the json more readable.
                result.update(obj.__dict__["__values__"])
            else:
                result.update(obj.__dict__)
            return result
        raise Exception("JSON serialisation not implemented yet for: " +
                        str(obj) + " type " + str(type(obj)))


def save_traces_to_json(traces, filename) -> None:
    # TODO: include signature and other meta-data (rand seed etc.) at the start of the file.
    with open(filename, "w") as output:
        json.dump(traces, output, indent=2, cls=TraceEncoder)
#        output.write("[")
#        for tr in traces:
#            comma_outer = "," if i > 0 else ""
#            output.write(f"{comma_outer}\n  [")
#            tr_num = 0
#            for op in tr:
#                comma_inner = "," if tr_num > 0 else ""
#                output.write(f"{comma_inner}\n    ")
#                output.write(jsonpickle.encode(op))
#                tr_num += 1
#            output.write("\n  ]")
#        output.write("\n]\n")


def load_traces_from_json(filename) -> List[List]:
    with open(filename, "r") as input:
        return json.load(input)
