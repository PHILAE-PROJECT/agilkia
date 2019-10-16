# -*- coding: utf-8 -*-
"""
Data structures for Traces and Sets of Traces.

This defines the 'Trace' and 'TraceSet' classes, plus helper functions.

NOTE: private data fields (starting with '_') will not be stored in the JSON files.
    For example, each Trace object has a '_parent' point to its TraceSet, but this
    is not stored in the JSON file, since the hierarchical structure of the JSON
    already captures the parent-child relationship between TraceSet and Trace.

NOTE: file version upgrade policy:
    JSON trace file version numbers follow the usual Semantic Versioning scheme:
        (Major.Minor.Patch).
    TraceSet.upgrade_json_data' currently just prints a warning message when a
    program running older code reads a JSON file with a newer MINOR version number.
    This allows graceful updating of one program at a time, but does
    have the danger that a older program may read newer data (with a warning),
    then SAVE that data in the slightly older format, thus losing some data.
    But a strict version-equality means that all programs have to be updated
    simultaneously, which is a pain.

TODO:
    * DONE: add save_as_arff() method like to_pandas.
    * DONE: store event_chars into meta_data.
    * DONE: store signatures into meta_data.
    * split RandomTester into SmartTester subclass (better meta-data).
    * add ActionChars class?
    * extend to_pandas() to allow user-defined columns to be added.
    * add support for clustering traces and visualising the clusters.

@author: utting@usc.edu.au
"""

import os
import sys
from pathlib import Path  # object-oriented filenames!
import json
import decimal
import datetime
import re
import xml.etree.ElementTree as ET
import pandas as pd
import arff   # liac-arff from https://pypi.org/project/liac-arff
from typing import List, Set, Mapping, Dict, Union


# Define some type synonyms
# =========================
# An event is a dictionary that maps string keys to either a string or a nested dictionary.
# Every event has at least these keys: "action":str, "inputs":dict, "outputs":dict
Event = Dict[str, Union[str, Mapping[str, str]]]


TRACE_SET_VERSION = "0.1.2"


def safe_name(string: str) -> str:
    """Returns 'string' with all non-alpha-numeric characters replaced by '_'."""
    return re.sub("[^A-Za-z0-9]", "_", string)


class Trace:
    """Represents a single trace, which contains a sequence of events.
    """

    def __init__(self, events: List[Event], parent: 'TraceSet' = None, random_state=None):
        """Create a Trace object from a list of events.

        Args:
            events: the sequence of Events that make up this trace.
            parent: the TraceSet that this trace is part of.
            random_state: If this trace was generated using some randomness, you should supply
                this optional parameter, to record the state of the random generator at the
                start of the sequence.  For example, rand_state=rand.getstate().
        """
        self.events = events
        self._parent = parent
        self.random_state = random_state

    def trace_set(self):
        """Returns the TraceSet that this trace is part of, or None if not known."""
        return self._parent

    def __iter__(self):
        return self.events.__iter__()

    def to_string(self,
                  to_char: Mapping[str, str] = None,
                  compress: List[str] = None,
                  color_status: bool = False):
        """Return a one-line summary of this trace, one character per event.
        See 'trace_to_string' for details.
        NOTE: throws an exception if no to_char map is given and this trace has no parent.
        """
        if to_char is None:
            if self._parent is None:
                raise Exception("Cannot view trace with no parent and no to_char map.")
            to_char = self._parent.get_event_chars()
        return trace_to_string(self.events, to_char, compress=compress, color_status=color_status)

    def __str__(self):
        try:
            return self.to_string()
        except Exception:
            return "..."


class TraceSet:
    """Represents a set of traces, either generated or recorded.

    Invariants:
        * forall tr:self.traces (tr._parent is self)
          (TODO: set _parent to None when a trace is removed?)
        * self.meta_data is a dict with keys: date, source at least.
    """

    def __init__(self, traces: List[Trace], meta_data: Mapping[str, str] = None):
        """Create a TraceSet object from a list of Traces.

        Args:
            traces: the list of Traces.
            meta_data: a dictionary that captures the important meta-data for this set of traces.
                If this TraceSet is to be saved into a file, the meta-data should include at
                least the GDF (General Data Format) compulsory fields,
                which are based on the Dublin Core:
                    "date" in ISO 8601 format;
                    "dataset" for the official name of this TraceSet;
                    "source" for the origin of the dataset;
                    and any other meta-data that is available.
        """
        self.version = TRACE_SET_VERSION
        if meta_data is None:
            meta_data = self.get_default_meta_data()
        self.meta_data = meta_data
        self.traces = traces
        for tr in self.traces:
            if isinstance(tr, Trace):
                tr._parent = self
            else:
                raise Exception("TraceSet expects List[Trace], not: " + str(type(tr)))
        self._event_chars = None  # recalculated if set of traces grows.

    def __iter__(self):
        return self.traces.__iter__()

    @classmethod
    def get_default_meta_data(cls):
        """Generates some basic meta-data such as date, user and command line."""
        now = datetime.datetime.now().isoformat()
        user = os.path.expanduser('~').split('/')[-1]  # usually correct, but can be tricked.
        meta_data = {"date": now, "author": user, "dataset": "unknown", "action_chars": None}
        if len(sys.argv) > 0:
            meta_data["source"] = sys.argv[0]  # the path to the running script/tool.
            meta_data["cmdline"] = sys.argv
        return meta_data

    def get_meta(self, key: str) -> any:
        """Returns requested meta data, or None if that key does not exist."""
        if key in self.meta_data:
            return self.meta_data[key]
        else:
            return None

    def append(self, trace: Trace):
        """Appends the given trace into this set.
        This also sets its parent to be this set.
        """
        trace._parent = self
        self.traces.append(trace)
        self._event_chars = None  # we will recalculate this later

    def set_event_chars(self, given: Mapping[str, str] = None):
        """Sets up the event-to-char map that is used to visualise traces.

        This will calculate a default mapping for any actions that are not in given.
        (See 'default_map_to_chars').

        Args:
            given: optional pre-allocation of a few action names to chars.
            For good readability of the printed traces, it is recommended that extremely
            common actions should be mapped to 'small' characters like '.' or ','.
        """
        if given is not None:
            self.meta_data["action_chars"] = given  # override any previous given map.
        else:
            given = self.get_meta("action_chars")
        actions = all_action_names(self.traces)
        self._event_chars = default_map_to_chars(actions, given=given)

    def get_event_chars(self):
        """Gets the event-to-char map that is used to visualise traces.
        This maps each action name to a single character.
        If set_event_chars has not been called, this getter will calculate and cache
        a default mapping from action names to characters.
        """
        if self._event_chars is None:
            self.set_event_chars()
        return self._event_chars

    def __str__(self):
        name = self.meta_data["dataset"]  # required meta data
        return f"TraceSet '{name}' with {len(self.traces)} traces."

    def save_to_json(self, file: Path) -> None:
        if isinstance(file, str):
            print(f"WARNING: converting {file} to Path.  Please learn to speak pathlib.")
            file = Path(file)
        with file.open("w") as output:
            json.dump(self, output, indent=2, cls=TraceEncoder)

    @classmethod
    def load_from_json(cls, file: Path) -> 'TraceSet':
        if isinstance(file, str):
            print(f"WARNING: converting {file} to Path.  Please learn to speak pathlib.")
            file = Path(file)
        if not isinstance(file, Path):
            raise Exception(f"load_from_json requires Path, not {file} (type={type(file)})")
        # with open(filename, "r") as input:
        data = json.loads(file.read_text())
        # Now check version and upgrade if necessary.
        if isinstance(data, list):
            # this file was pre-TraceSet, so just a list of lists of events.
            mtime = datetime.datetime.fromtimestamp(file.stat().st_mtime).isoformat()
            meta = {"date": mtime, "dataset": file.name, "source": "Upgraded from version 0.1"}
            traces = cls([], meta)
            for ev_list in data:
                traces.append(Trace(ev_list))
            return traces
        elif isinstance(data, dict) and data.get("__class__", None) == "TraceSet":
            return cls.upgrade_json_data(data)
        else:
            raise Exception("unknown JSON file format: " + str(data)[0:60])

    @classmethod
    def upgrade_json_data(cls, json_data: Dict) -> 'TraceSet':
        version = json_data["version"]
        if version.startswith("0.1."):
            # This JSON file is compatible with our code.
            # First, convert json_data dicts to Trace and TraceSet objects.
            traceset = TraceSet([], json_data["meta_data"])
            for tr_data in json_data["traces"]:
                assert tr_data["__class__"] == "Trace"
                rand = tr_data.get("random_state", None)
                traceset.append(Trace(tr_data["events"], random_state=rand))
            # Next, see if any little updates are needed.
            if version == TRACE_SET_VERSION:
                pass  # nothing extra to do.
            elif version == "0.1.1":
                # Move given_event_chars into meta_data["action_chars"]
                # Note: traceset["version"] has already been updated to the latest.
                traceset.meta_data["actions_chars"] = json_data["given_event_chars"]
            else:
                # The JSON must be from a newer 0.1.x version, so give a warning.
                print(f"WARNING: reading {version} TraceSet using {TRACE_SET_VERSION} code.")
                print(f"         Some data may be lost.  Please upgrade this program.")
            return traceset
        raise Exception(f"upgrade of TraceSet v{version} to v{TRACE_SET_VERSION} not supported.")

    def to_pandas(self) -> pd.DataFrame:
        """Converts all the traces into a single Pandas DataFrame (one event/row).

        The first three columns are 'Trace' and 'Event' which give the number of the
        trace and the position of the event within that trace, and 'Action' which is
        the name of the action of the event.
        Each named input value is recorded in a separate column.
        For outputs, by default there are just 'Status' (int) and 'Error' (str) columns.
        """
        return traces_to_pandas(self.traces)

    def arff_type(self, pandas_type: str) -> str:
        """Maps each Pandas data type to the closest ARFF type."""
        if pd.api.types.is_integer_dtype(pandas_type):
            return "INTEGER"
        if pd.api.types.is_float_dtype(pandas_type):
            return "REAL"
        if pd.api.types.is_bool_dtype(pandas_type):
            return ["False", "True"]
        return "STRING"
        # TODO: check column to see if NOMINAL is better?
        # raise Exception(f"do not know how to translate Pandas type {pandas_type} to ARFF.")

    def save_to_arff(self, file: Path, name=None) -> None:
        """Save all the events in all traces into an ARFF file for machine learning.

        Args:
            filename: the name of the file to save into.  Should end with '.arff'.
            name: optional relation name to identify this data inside the ARFF file.
                The default is the base name of 'file'.
        """
        if isinstance(file, str):
            print(f"WARNING: converting {file} to Path.  Please learn to speak pathlib.")
            file = Path(file)
        if name is None:
            name = file.stem
        data = self.to_pandas()
        attributes = [(n, self.arff_type(t)) for (n, t) in zip(data.columns, data.dtypes)]
        with file.open("w") as output:
            contents = {
                "relation": safe_name(name),
                "attributes": attributes,
                "data": data.values,  # [[tr] for tr in trace_summaries],
                "description": "Events from " + name
                }
            arff.dump(contents, output)


class TraceEncoder(json.JSONEncoder):
    """Custom JSON encoder because objects from zeep could not be serialised.

    Based on ideas from this blog entry by 'The Fellow' (Ouma Rodgers):
    https://medium.com/python-pandemonium/json-the-python-way-91aac95d4041.

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
                self._add_public_attributes(result, obj.__dict__["__values__"])
            else:
                self._add_public_attributes(result, obj.__dict__)
            return result
        raise Exception("JSON serialisation not implemented yet for: " +
                        str(obj) + " type " + str(type(obj)) + " dir:" + ",".join(dir(obj)))

    def _add_public_attributes(self, result, attrs) -> None:
        for (name, value) in attrs.items():
            if not name.startswith("_"):
                result[name] = value


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


def default_map_to_chars(actions: List[str], given: Mapping[str, str] = None) -> Mapping[str, str]:
    """Tries to guess a useful default mapping from action names to single characters.

    Args:
        actions: the names of all the actions.
        given: optional pre-allocation of a few action names to chars.
            You can use this to override the default behaviour.

    Returns:
        A map from every name in actions to a unique single character.
    """
    names = sorted(actions)
    result = {} if given is None else given
    # TODO: a better algorithm might be to break up compound words and look for word prefixes?
    curr_prefix = ""
    pass2 = []
    for i in range(len(names)):
        name = names[i]
        if name in result:
            continue  # given
        # skip over any prefix that was in common with previous name.
        if name.startswith(curr_prefix):
            pos = len(curr_prefix)
        else:
            pos = 0
        # check ahead for common prefixes first
        if i + 1 < len(names):
            nxt = names[i + 1]
            if nxt.startswith(name) and name[0] not in result.values():
                result[name] = name[0]
                curr_prefix = name
                continue
            prefix = max([p for p in range(max(len(name), len(nxt))) if name[0:p] == nxt[0:p]])
            # print(f"  found prefix {prefix} of {name} and {nxt}")
            curr_prefix = name[0:prefix]
        else:
            prefix = 0
            curr_prefix = ""
        if prefix > 0 and prefix > pos:
            pos = prefix
        done = False
        for j in range(pos, len(name)):
            if name[pos] not in result.values():
                result[name] = name[pos]
                done = True
                break
        if not done:
            pass2.append(name)
    # Pass 2 (all visible ASCII chars except " and ')
    allchars = "".join([chr(n) for n in range(42, 127)]) + "!#$%&()"
    for name in pass2:
        for ch in name + allchars:
            if ch not in result.values():
                result[name] = ch
                break  # move onto next name in pass2
    return result


def all_action_names(traces: List[Trace]) -> Set[str]:
    """Collects all the action names that appear in the given traces."""
    result = set()
    for tr in traces:
        for ev in tr.events:
            action = ev["action"]
            result.add(action)
    return result


def event_status(event: Event) -> int:
    """Get the status result for the given event."""
    return int(event["outputs"]["Status"])


def trace_to_string(trace: List[Event], to_char: Mapping[str, str], compress: List[str] = None,
                    color_status: bool = False) -> str:
    """Converts a trace to a short summary string, one character per action.

    Args:
        trace: the sequence of JSON-like events, with an "action" field.
        to_char: maps each action name to a single character.  This map must include every
            action name that appears in the traces.  A suitable map can be constructed via
            TraceSet.get_event_chars().
        compress: a list of Action names.  Repeated events will be compressed if in this list.
        color_status: True means color the string red where status is non-zero.
            This uses ANSI escape sequences, so needs to be printed to a terminal.

    Returns:
        a summary string.
    """
    compress_set = set() if compress is None else set(compress)
    chars = []
    prev_action = None
    for ev in trace:
        action = ev["action"]
        if action == prev_action and action in compress_set:
            # NOTE: we color compressed output just based on the first event.
            pass
        else:
            if color_status and event_status(ev) != 0:
                chars.append("\033[91m")  # start RED
                chars.append(to_char[action])
                chars.append("\033[0m")  # turn off color
            else:
                chars.append(to_char[action])
            prev_action = action
    return "".join(chars)


def traces_to_pandas(traces: List[Trace]) -> pd.DataFrame:
    """Collects all events into a single Pandas DataFrame.

    Columns include the trace number, the event number, the action name, each input parameter,
    the result status and error message.

    TODO: we could convert complex values to strings before sending to Pandas?
    """
    rows = []
    for tr_num in range(len(traces)):
        events = traces[tr_num].events
        for ev_num in range(len(events)):
            event = events[ev_num]
            row = {"Trace": tr_num, "Event": ev_num, "Action": event["action"]}
            # we add "Status" and "Error" first, so that those columns come before inputs.
            row["Status"] = event_status(event)
            row["Error"] = event["outputs"].get("Error", None)
            row.update(event["inputs"].items())
            rows.append(row)
    return pd.DataFrame(rows)
