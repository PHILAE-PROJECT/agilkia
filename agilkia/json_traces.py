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
import xml.etree.ElementTree as ET
from typing import List, Mapping, Union


class TraceEncoder(json.JSONEncoder):
    """Custom JSON encoder because objects from zeep could not be serialised.

    This does not handle XML objects, as they should be decoded via xml_decode first.
    """

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
            if len(obj.__dict__) == 1 and "__values__" in obj.__dict__:
                # zeep seems to hide the attributes in a __values__ dict.
                # We lift them up to the top level to make the json more readable.
                result.update(obj.__dict__["__values__"])
            else:
                result.update(obj.__dict__)
            return result
        raise Exception("JSON serialisation not implemented yet for: " +
                        str(obj) + " type " + str(type(obj)) + " dir:" + ",".join(dir(obj)))


def xml_decode(obj: ET.Element) -> Union[str, Mapping[str, any]]:
    """Custom XML encoder to decode XML into a Python dictionary suitable for JSON encoding.

    This roughly follows the ideas from:
    https://www.xml.com/pub/a/2006/05/31/converting-between-xml-and-json.html.

    For simple XML objects with no attributes and no children, this returns just the text string.
    For more complex XML objects, it returns a dictionary.

    Note that the top-level tag of 'obj' is assumed to be handled by the caller.
    That is, the caller will typically do ```d[tag] = xml_decode(obj)``` where xml_decode
    will return either a simple string, or a dictionary.
    """
    if len(obj) == 0 and len(obj.attrib) == 0:
        return obj.text
    else:
        # return obj as a dictionary
        result = {}
        for (n, v) in obj.attrib.items():
            result[n] = v
        # child objects are more tricky, since some tags may appear multiple times.
        # If a tag appears multiple times, we map it to a list of child objects.
        curr_tag = None
        curr_list = []
        for child in obj:
            if child.tag != curr_tag:
                # save the child(ren) we have just finished
                if len(curr_list) > 0:
                    result[curr_tag] = curr_list if len(curr_list) > 1 else curr_list[0]
                curr_list = []
                curr_tag = child.tag
            curr_list.append(xml_decode(child))
        if len(curr_list) > 0:
            result[curr_tag] = curr_list if len(curr_list) > 1 else curr_list[0]
        if obj.text and obj.text.strip():  # ignore text that is just whitespace
            result["text"] = obj.text
        return result


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
