"""
Simple random test generator for SOAP web services.

Author: Mark Utting, 2019

TODO:
* provide a way of generating related inputs like (lat,long) together.

"""


import csv
import requests
import zeep   # type: ignore
import zeep.helpers   # type: ignore
import getpass
import operator
import random
import numpy   # type: ignore
import unittest
from pathlib import Path
from pprint import pprint
from typing import Tuple, List, Mapping, Dict, Any, Optional, Union

from . json_traces import Event, Trace, TraceSet


# A signature of a method maps "input"/"output" to the dictionary of input/output names and types.
Signature = Mapping[str, Mapping[str, str]]
InputRules = Dict[str, List[str]]


# TODO: make these user-configurable
DUMP_WSDL = False         # save each *.wsdl file into current directory.
DUMP_SIGNATURES = False    # save summary of methods into *_signatures.txt
GOOD_PASSWORD = "<GOOD_PASSWORD>"


def read_input_rules(file: Path) -> InputRules:
    """Reads a CSV file of input values.

    The header line of the CSV file should contain headers: Name,Frequency,Value.
    (but the Frequency column is optional, and missing frequencies default to 1).

    For example if one line contains 'size,3,100' and another contains 'size,2,200',
    then the resulting input rules will define a 3/5 chance of size being 100,
    and a 2/5 chance of it being 200.
    """
    input_rules: InputRules = {}
    with open(file, "r") as input:
        for row in csv.DictReader(input):
            name = row["Name"]
            freq = row.get("Frequency", "")
            freq_int = int(freq) if freq else 1
            value = row["Value"]
            value_list = input_rules.get(name, [])
            for i in range(freq_int):
                value_list.append(value)
            input_rules[name] = value_list  # update it after appending new values
    print(input_rules)
    return input_rules


def summary(value) -> str:
    """Returns a one-line summary of the given value."""
    s = str(value).replace("\n", "").replace(" ", "")
    return s[:60]


def uniq(d):
    """Returns the unique value of a dictionary, else an empty dictionary."""
    result = {}
    for k, v in d.items():
        if result == {} or result == v:
            result = v
            return result  # temp hack - ITM 3 ports have slight differences.
        else:
            print(f"WARNING: uniq sees different values.\n" +
                  " val1={result}\n  val2={v}")
            return {}
    return result


class TestUniq(unittest.TestCase):
    """Some unit tests of the uniq function."""

    def test_normal(self):
        self.assertEqual("def", uniq({"abc": "def"}))

    # TODO: assert uniq({"abc":"one", "xyz":"two"}) == {}

    def test_duplicate_values(self):
        self.assertEquals("one", uniq({"abc": "one", "xyz": "one"}))


def parse_elements(elements):
    """Helper function for build_interface."""
    all_elements = {}
    for name, element in elements:
        all_elements[name] = {}
        all_elements[name]['optional'] = element.is_optional
        if hasattr(element.type, 'elements'):
            all_elements[name]['type'] = parse_elements(element.type.elements)
        else:
            all_elements[name]['type'] = str(element.type)
    return all_elements


def build_interface(client: zeep.Client) -> Dict[str, Dict[str, Any]]:
    """Returns a nested dictionary structure for the methods of client.

    Typical usage to get a method called "Login" is:
    ```build_interface(client)[service][port]["operations"]["Login"]```
    """
    interface: Dict[str, Dict[str, Any]] = {}
    for service in client.wsdl.services.values():
        interface[service.name] = {}
        for port in service.ports.values():
            interface[service.name][port.name] = {}
            operations: Dict[str, Any] = {}
            for operation in port.binding._operations.values():
                operations[operation.name] = {}
                operations[operation.name]['input'] = {}
                elements = operation.input.body.type.elements
                operations[operation.name]['input'] = parse_elements(elements)
            interface[service.name][port.name]['operations'] = operations
    return interface


def print_signatures(client: zeep.Client, out):
    """Print a short summary of each operation signature offered by client."""
    # From: https://stackoverflow.com/questions/50089400/introspecting-a-wsdl-with-python-zeep
    for service in client.wsdl.services.values():
        out.write(f"service: {service.name}\n")
        for port in service.ports.values():
            out.write(f"  port: {port.name}\n")
            operations = sorted(
                port.binding._operations.values(),
                key=operator.attrgetter('name'))
            for operation in operations:
                action = operation.name
                inputs = operation.input.signature()
                outputs = operation.output.signature()
                out.write(f"    {action}({inputs})  --> ({outputs})\n")


class RandomTester:
    """Does random testing of a given web service.

    Give it a URL to a web service (or a list of URLs if there are several web services),
    and it will read the WSDL specifications from those web services and
    generate any number of random test sequences to test the methods.

    For more sophisticated (user-directed) testing you can also:
    * supply a username and password if login credentials are needed.
    * supply the subset of method names that you want to focus on testing (default is all).
    * supply a set of default input values (or generation functions) for each data type.
    * supply a set of input values (or generation functions) for each named input parameter.
    """
    def __init__(self,
                 urls: Union[str, List[str]],
                 methods_to_test: List[str] = None,
                 input_rules: Dict[str, List] = None,
                 rand: random.Random = None,
                 action_chars: Mapping[str, str] = None,
                 verbose: bool = False):
        """Creates a random tester for the server url and set of web services on that server.

        Args:
            urls (str or List[str]): URLs to the web services, used to find the WSDL files.
            methods_to_test (List[str]): only these methods will be tested (None means all).
            input_rules (Dict[str,List]): maps each input parameter name to a list of
                possible values, one of which will be chosen randomly.
            rand (random.Random): the random number generator used to generate tests.
            action_chars (Mapping[str, str]): optional action-to-character map, for visualisation.
            verbose (bool): True means print progress messages during test generation.
        """
        self.urls = [urls] if isinstance(urls, str) else urls
        self.username: Optional[str] = None
        self.password: Optional[str] = None
        self.random = random.Random() if rand is None else rand
        self.verbose = verbose
        self.clients_and_methods: List[Tuple[zeep.Service, Dict[str, Signature]]] = []
        self.methods_to_test = methods_to_test
        self.methods_allowed = [] if methods_to_test is None else methods_to_test
        # maps each parameter to list of possible 'values'
        self.named_input_rules = {} if input_rules is None else input_rules
        meta = TraceSet.get_default_meta_data()
        meta["source"] = "RandomTester"
        meta["web_services"] = self.urls
        meta["methods_to_test"] = methods_to_test
        meta["input_rules"] = input_rules
        meta["method_signatures"] = {}  # see add_web_service
        meta["action_chars"] = action_chars
        new_trace = Trace([], random_state=self.random.getstate())
        self.curr_events = new_trace.events  # mutable list to append to.
        self.trace_set = TraceSet([], meta)
        self.trace_set.append(new_trace)
        for w in self.urls:
            self.add_web_service(w)

    def set_username(self, username: str, password: str = None):
        """Set the username and (optional) password to be used for the subsequent operations.
        If password is not supplied, this method will immediately interactively prompt for it.
        """
        self.username = username
        self.trace_set.meta_data["username"] = username
        self.password = password or getpass.getpass(f"Please enter password for user {username}:")

    def add_web_service(self, url: str):
        """Add another web service using the given url."""
        wsdl = url + ("" if url.upper().endswith("WSDL") else ".asmx?WSDL")
        name = url.split("/")[-1]
        print("  loading WSDL: ", wsdl)
        if DUMP_WSDL:
            # save the WSDL for reference
            r = requests.get(wsdl, allow_redirects=True)
            open(f"{name}.wsdl", 'wb').write(r.content)
        # now create the client interface for this web service
        client = zeep.Client(wsdl=wsdl)
        interface = build_interface(client)
        pprint([(k, len(v["operations"])) for k, v in uniq(interface).items()])
        if DUMP_SIGNATURES:
            # save summary of this web service into a signatures file
            with open(f"{name}_signatures.txt", "w") as sig:
                print_signatures(client, sig)
        if not uniq(interface):
            print(f"WARNING: web service {name} has empty interface?")
            pprint(interface)
        else:
            ops = uniq(uniq(interface))["operations"]
            self.clients_and_methods.append((client, ops))
            self.trace_set.meta_data["method_signatures"].update(ops)
            if self.methods_to_test is None:
                self.methods_allowed += list(ops.keys())

    def _find_method(self, name: str) -> Tuple[zeep.Client, Signature]:
        """Find the given method in one of the web services and returns its signature."""
        for (client, interface) in self.clients_and_methods:
            if name in interface:
                return client, interface[name]
        raise Exception(f"could not find {name} in any WSDL specifications.")

    def choose_input_value(self, arg_name: str) -> str:
        """Choose an appropriate value for the input argument called 'arg_name'.
        If no set of input rules is defined for 'arg_name', then 'generate_input_value'
        is called to generate a suitable input value.  Subclasses can override this.

        Args:
            arg_name (str): the name of the input parameter.

        Returns:
            a string if successful, or None if no suitable value was found.
        """
        values = self.named_input_rules.get(arg_name, None)
        if values is None:
            return self.generate_input_value(arg_name)
        val = self.random.choice(values)
        return val

    def generate_input_value(self, arg_name: str) -> Any:
        """Can be overridden in subclasses to generate smart values for an input argument."""
        print(f"ERROR: please define possible parameter values for input {arg_name}")
        return None

    def _insert_password(self, arg_value: str) -> str:
        if arg_value == GOOD_PASSWORD:
            if self.password is None:
                raise Exception("Please call set_username before using " + GOOD_PASSWORD)
            return self.password
        else:
            return arg_value

    def get_methods(self) -> Mapping[str, Signature]:
        """Return the set of all method names in all the web services."""
        methods = {}
        for (client, interface) in self.clients_and_methods:
            methods.update(interface)
        return methods

    def call_method(self, name: str, args: Dict[str, Any] = None):
        """Call the web service name(args) and add the result to trace.

        Args:
            name (str): the name of the method to call.
            args (dict): the input values for the method.  If args=None, then this method uses
                'choose_input_value' to choose appropriate values for each argument
                value of the method.
        Returns:
        Before the call, this method replaces some symbolic inputs by actual concrete values.
        For example the correct password token is replaced by the real password --
        this avoids recording the real password in the inputs of the trace.

        Returns:
            all the data returned by the method.
        """
        (client, signature) = self._find_method(name)
        inputs = signature["input"]
        if args is None:
            args = {n: self.choose_input_value(n) for n in inputs.keys()}
        if None in args.values():
            print(f"skipping method {name}.  Please define missing input values.")
            return None
        if self.verbose:
            print(f"    call {name}{args}")
        # insert special secret argument values if requested
        args_list = [self._insert_password(arg) for (n, arg) in args.items()]
        out = getattr(client.service, name)(*args_list)
        # we call it 'action' so it gets printed before 'inputs' (alphabetical order).
        self.curr_events.append(Event(name, args, out))
        if self.verbose:
            print(f"    -> {summary(out)}")
        return out

    def generate_trace(self, start=True, length=20, methods: List[str] = None) -> Trace:
        """Generates the requested length of test steps, choosing methods at random.

        Args:
            start (bool): True means that a new trace is started (unless current one is empty).
            length (int): The number of steps to generate (default=20).
            methods (List[str]): only these methods will be chosen (None means all are allowed)

        Returns:
            the whole of the current trace that has been generated so far.
        """
        if start:
            if len(self.curr_events) > 0:
                new_trace = Trace([], random_state=self.random.getstate())
                self.curr_events = new_trace.events  # mutable list to append to.
                self.trace_set.append(new_trace)
        if methods is None:
            methods = self.methods_allowed
        for i in range(length):  # TODO: continue while Status==0?
            self.call_method(self.random.choice(methods))
        return self.trace_set.traces[-1]

    def setup_feature_data(self):
        """Must be called before the first call to get_trace_features."""
        actions = self.methods_allowed
        nums = len(actions)
        self.action2number = dict(zip(actions, range(nums)))
        if self.verbose:
            print("Action 2 num:", self.action2number)

    def get_action_counts(self, events: List[Event]) -> List[int]:
        """Returns an array of counts - how many times each event occurs in trace."""
        result = [0 for k in self.action2number.keys()]
        for ev in events:
            action_num = self.action2number[ev.action]
            result[action_num] += 1
        return result

    def get_trace_features(self) -> List[int]:
        """Returns a vector of numeric features suitable for input to an ML model.
        The results returned by this function must match the training set of the ML model.
        Currently this returns an array of counts - how many times each event occurs
        in the whole current trace, and how many times in the most recent 8 events.
        """
        prefix = self.get_action_counts(self.curr_events)
        suffix = self.get_action_counts(self.curr_events[-8:])
        return prefix+suffix

    def generate_trace_ml(self, model, start=True, length=20):
        """Generates the requested length of test steps, choosing methods using the given model.

        Args:
            model (Any): the ML model to use to generate the next event.
                This model must support the 'predict_proba' method.
            start (bool): True means that a new trace is started, beginning with a "Login" call.
            length (int): The number of steps to generate (default=20).

        Returns:
            the whole of the current trace that has been generated so far.
        """
        self.setup_feature_data()
        # start a new (empty) trace if requested.
        self.generate_trace(start=start, length=0)
        for i in range(length):
            features = self.get_trace_features()
            [proba] = model.predict_proba([features])
            [action_num] = numpy.random.choice(len(proba), p=proba, size=1)
            action = self.methods_allowed[action_num]
            if self.verbose:
                print(i, features, action, ",".join([f"{int(p*100)}" for p in proba]))
            self.call_method(action)
        return self.curr_events


if __name__ == "__main__":
    unittest.main()
