"""
Simple random test generator for SOAP web services.

Author: Mark Utting, 2019

TODO:
* provide a way of generating related inputs like (lat,long) together.
* improve SmartSequenceGenerator to pass scikit-learn estimator tests?
  See sklearn.utils.estimator_checks.check_estimator(Estimator)

"""


import csv
import requests
import zeep   # type: ignore
import zeep.helpers   # type: ignore
import getpass
import operator
import random
import collections
import unittest
import sklearn.base
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from pathlib import Path
from pprint import pprint
from typing import Tuple, List, Set, Mapping, Dict, Counter, Any, Optional, Union

from . json_traces import Event, Trace, TraceSet, TraceEncoder, MetaData


# A signature of a method maps "input"/"output" to the dictionary of input/output names and types.
Signature = Mapping[str, Mapping[str, str]]
InputRules = Dict[str, List[str]]


# TODO: make these user-configurable
DUMP_WSDL = False         # save each *.wsdl file into current directory.
DUMP_SIGNATURES = False    # save summary of methods into *_signatures.txt
GOOD_PASSWORD = "<GOOD_PASSWORD>"
TRACE_END = "<END>"


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
    """Some unit tests of the uniq function.

    (This should be in a test file, but uniq is not exported.)"""

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
                 method_signatures: Dict[str, Signature] = None,
                 methods_to_test: List[str] = None,
                 input_rules: Dict[str, List] = None,
                 rand: random.Random = None,
                 action_chars: Mapping[str, str] = None,
                 verbose: bool = False):
        """Creates a random tester for the server url and set of web services on that server.

        Args:
            urls: Optional list of WSDL URLs.  If this is the empty list, then
                method_signatures must be provided.
            method_signatures: Optional mapping from method names to input/output signatures.
                This will be inferred automatically if urls is provided.
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
        self.clients_and_methods: List[Tuple[Optional[zeep.Service], Dict[str, Signature]]] = []
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
        if self.urls:
            for w in self.urls:
                self.add_web_service(w)
        else:
            # We allow user-supplied dummy signatures when running offline (no web service)
            if method_signatures is None:
                raise Exception("urls or method_signatures must be provided")
            self.clients_and_methods.append((None, method_signatures))
            self.trace_set.meta_data["method_signatures"].update(method_signatures)
        if self.methods_to_test is None and method_signatures is not None:
            self.methods_allowed += sorted(list(method_signatures.keys()))

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

    def summary(self, value) -> str:
        """Returns a one-line summary of the given value."""
        s = str(value).replace("\n", "").replace(" ", "")
        return s[:95]

    def decode_outputs(self, raw) -> Dict[str, Any]:
        """Decode the outputs from a web service/site call into a dictionary.
        
        This adds a 'Status' entry in the output dictionary,
        to say if the operation was successful (0) or not (non-zero).
        """
        # Since raw comes from a zeep call, it should be XML.
        if isinstance(raw, dict):
            out = raw.copy()
        elif isinstance(raw, str):
            out = {"value": raw}
        else:
            out = TraceEncoder().default(raw)
            if not isinstance(out, dict):
                out = {"value": out}
        if "Status" not in out:
            out["Status"] = 0
        return out

    def call_method(self, name: str, args: Dict[str, Any] = None,
                    meta_data: Optional[MetaData] = None):
        """Call the web service name(args) and add the result to trace.

        Args:
            name (str): the name of the method to call.
            args (dict): the input values for the method.  If args=None, then this method uses
                'choose_input_value' to choose appropriate values for each argument
                value of the method.
            meta_data: optional meta data to add to the resulting Event.
        Returns:
        Before the call, this method replaces some symbolic inputs by actual concrete values.
        For example the correct password token is replaced by the real password --
        this avoids recording the real password in the inputs of the trace.

        Returns:
            The whole Event object created by this method call.
        """
        (client, signature) = self._find_method(name)
        inputs = signature["input"]
        if args is None or (len(args) == 0 and len(inputs) > 0):
            args = {n: self.choose_input_value(n) for n in inputs.keys()}
        if None in args.values():
            print(f"skipping method {name}.  Please define missing input values.")
            return None
        if self.verbose:
            print(f"    call {name}{args}")
        if client is None:
            out = {"Status": 0}  # dummy results, always succeeds.
        else:
            # insert special secret argument values if requested
            args_list = [self._insert_password(arg) for (n, arg) in args.items()]
            raw_out = getattr(client.service, name)(*args_list)
            out = self.decode_outputs(raw_out)
        event = Event(name, args, out, meta_data=meta_data)
        self.curr_events.append(event)
        if self.verbose:
            print(f"      -> {self.summary(event.outputs)}")
        return event

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


class TracePrefixExtractor(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Encodes all the prefixes of all traces into (X,y) features for machine learning.

    This feature encoder follows the standard scikit-learn Estimator conventions, so a
    typical usage might look like::

        prefixes = TracePrefixExtractor()
        prefixes.fit(traceset)
        X = prefixes.transform(traceset)
        y = prefixes.get_labels()

    The default implementation uses bag-of-words to build X.

    That is, each event is converted into a single string (see `event_to_str` parameter),
    and then bag-of-words is used to count the number of times each of those strings appears.
    This gives one row of the X matrix, while the corresponding y value (see `get_labels()`)
    is just the result of applying event_to_str to the next event.

    The default event_to_str function just uses the action string for each Event, which
    is useful for learning to predict the next action.  If 'tr' is the first Trace in the
    traceset in the above example, then the first few rows in (X,y) will effectively be::

        X[i] = Counter([ev.action for ev in tr[0:i]])
        y[i] = tr[i].action

    For more complex feature encoding than simple bag-of-words, you can create a subclass
    of this class and override the `generate_row` and `generate_feature_names` methods.
    
    Attributes:
        vocabulary_ (dict): A dictionary mapping feature names to feature indices.
        feature_names_ (list): A list of length n_features containing the feature names.
    """

    def __init__(self, event_to_str=None):
        """
        Create a new prefix-extractor for traces.

        Args:
            event_to_str (Event -> str): an optional feature-extractor function that maps each event
               to a single string.  The default is just to return the action name of the event.
        """
        super().__init__()
        self._traceset = None
        self.is_fitted_ = False
        if event_to_str is None:
            self._event_to_str = (lambda ev: ev.action)
        else:
            self._event_to_str = event_to_str

    def get_feature_names(self):
        """Gets the list of column names for the generated data tables."""
        check_is_fitted(self, 'is_fitted_')
        return self.feature_names_

    def set_feature_names(self, names: List[str]):
        """Sets the output column names to the given list of feature names.

        Also calculates the inverse mapping (names to position) for internal use.

        Args:
            names: For consistent results, this list should be sorted in a consistent way.
        """
        self.feature_names_ = names
        self.vocabulary_ = {(col, i) for (i, col) in enumerate(self.feature_names_)}

    def generate_feature_names(self, trace: Trace) -> Set[str]:
        """Generate the column names required for the given trace.

        By default this just applies the event_to_str function to every Event in trace.
        """
        return {self._event_to_str(ev) for ev in trace}

    def get_prefix_features(self, events: List[Event]) -> Dict[str, float]:
        """Use bag-of-words to count the various features in a sequence of Events."""
        # Note: Counter is a dictionary, but mypy does not recognise this.
        return collections.Counter([self._event_to_str(ev) for ev in events])

    def generate_prefix_features(self, events: List[Event], current: Optional[Event]) -> Tuple[Dict[str, float], Any]:
        """Encodes a sequence of events into one row of the (X,y) training data.
        
        Subclasses can override this to change the feature encoding (X), or change what is being learned (y).
        If they want to change the feature names, they should also override generate_feature_names.

        Args:
            events: the prefix of the trace.
            current: the next event in the trace.  None usually means the end of the trace.
        """
        counts = self.get_prefix_features(events)
        if current is None:
            expect = TRACE_END
        else:
            expect = self._event_to_str(current)
        return counts, expect

    def fit(self, X: TraceSet, y=None):
        """Fit uses the given TraceSet to calculate the feature names.

        It takes the union of `generate_feature_names()` over all the traces,
        sorts the resulting set of feature names, and passes that list to set_feature_names.

        Args:
            X (TraceSet): the set of traces to be fitted.
            y (None): unused.

        Returns:
            self

        Note that fit() must be called before transform() or get_feature_names().
        """
        if not isinstance(X, TraceSet):
            raise Exception("TracePrefixExtractor.fit input must be TraceSet.")
        self._traceset = X
        features = set()
        for tr in X:
            features |= self.generate_feature_names(tr)
        self.set_feature_names(sorted(list(features)))
        self.is_fitted_ = True
        return self

    def transform(self, X: Union[TraceSet, List[Event]], curr: Event = None):
        """Transforms a set of traces, or an event sequence, into a Pandas DataFrame
        of training data.  Note that the columns of the resulting DataFrame are fixed
        during the fit() method (which calls `set_feature_names`), so any new kinds
        of actions appearing in this X input will be ignored.

        There are two different behaviors, depending upon the input type of X.
            * if traces is a TraceSet, all prefixes of all traces are converted into
              training data, and the corresponding expected y labels (e.g. action name) for
              all those prefixes are available from get_labels().
            * if traces is a list of events, then the result will contain just
              a single row, which will be the data for that whole trace.  In this case,
              the optional parameter `curr` may be used to pass the partially-complete
              current event to the feature encoding if desired.
        """
        check_is_fitted(self, 'is_fitted_')
        data = []
        self._y = []
        if isinstance(X, TraceSet):
            for tr in X.traces:
                for size in range(len(tr) + 1):  # every prefix, plus the whole trace.
                    prefix = tr.events[0:size]
                    current = tr.events[size] if size < len(tr) else None
                    (row, y) = self.generate_prefix_features(prefix, current)
                    data.append(row)
                    self._y.append(y)
        elif isinstance(X, list) and (X == [] or isinstance(X[0], Event)):
            (row, y) = self.generate_prefix_features(X, curr)
            data = [row]
            self._y = [y]
        else:
            raise Exception("TracePrefixExtractor.transform input must be TraceSet or [Events].")
        df = pd.DataFrame(data, columns=self.feature_names_)
        df.fillna(0, inplace=True)
        return df

    def get_labels(self):
        """Get the output labels (action names) corresponding to the last transform() call."""
        check_is_fitted(self, 'is_fitted_')
        return self._y


class SmartSequenceGenerator(RandomTester):
    """Generates test sequences from an ML model that suggests what actions can come next."""
    
    def __init__(self,
                 urls: Union[str, List[str]],
                 method_signatures: Dict[str, Signature] = None,
                 methods_to_test: List[str] = None,
                 input_rules: Dict[str, List] = None,
                 rand: random.Random = None,
                 action_chars: Mapping[str, str] = None,
                 verbose: bool = False):
        """A test sequence generator that uses a machine learning model to predict next action.
        
        Args:
            urls: Optional list of WSDL URLs.  If this is the empty list, then
                method_signatures must be provided.
            method_signatures: Optional mapping from method names to input/output signatures.
                This will be inferred automatically if urls is provided.
            methods_to_test (List[str]): only these methods will be tested (None means all).
            input_rules (Dict[str,List]): maps each input parameter name to a list of
                possible values, one of which will be chosen randomly.
            rand (random.Random): the random number generator used to generate tests.
            action_chars (Mapping[str, str]): optional action-to-character map, for visualisation.
            verbose (bool): True means print progress messages during test generation.
        """
        super().__init__(urls, method_signatures=method_signatures, methods_to_test=methods_to_test, 
                        input_rules=input_rules, rand=rand, action_chars=action_chars, verbose=verbose)

    def generate_trace_with_model(self, model, start=True, length=20, event_factory=None):
        """Generates one sequence test steps, choosing actions using the given model.
        The generated trace terminates either when the model says <end> or after length steps.

        Args:
            model (Classifier): ML model that takes an Event list and predicts next action name.
            start (bool): True means that a new trace is started, beginning with a "Login" call.
            length (int): The maximum number of steps to generate in one trace (default=20).
            event_factory (string->Event): Optional event generator, from the string predicted by the model.

        Returns:
            the whole of the current trace that has been generated so far.
        """
        # start a new (empty) trace if requested.
        self.generate_trace(start=start, length=0)
        result = self.curr_events
        for i in range(length):
            [proba] = model.predict_proba(self.curr_events)
            [action_num] = self.random.choices(range(len(proba)), proba) 
            action = model.classes_[action_num] # WAS: self.methods_allowed[action_num]
            if self.verbose:
                probs = ",".join([f"{int(p*100)}" for p in proba])
                print(f"{i:3d}: {action:20s} {probs}")
            if action == TRACE_END:
                self.generate_trace(start=True, length=0)
                break  # we view length as a maximum...
            else:
                if event_factory is None:
                    self.call_method(action)
                else:
                    ev = event_factory(action)
                    # TODO: should this also do call_method?
                    self.curr_events.append(ev)
        return result

    def generate_all_traces(self, model, length=5, action_prob=0.01, path_prob=1.0e-12,
                            partial=True, event_factory=None) -> List[Trace]:
        """Generate all traces that satisfy the given constraints.

        Args:
            model: the trained ML model used to predict the next action.
            length (int): maximum length of each generated trace.
            action_prob (float): only do actions with at least this probability.
            path_prob (float): only include paths with at least this total probability.
            partial (bool): True means include partial traces.  False gives complete traces only.
            event_factory (string->Event): Optional event generator, from the string predicted by the model.

        Returns:
            A list of all the Trace objects that satisfy the given constraints.
                Note that all complete traces will have len(tr)<length, whereas all partial
                traces will have len(tr)==length.
        """
        results = []
        if event_factory is None:
            event_factory = (lambda action: Event(action,{},{}))
        def depth_first_search(prefix, prob):
            indent = ">" * len(prefix)
            # print(indent + ",".join([ev.action for ev in prefix]))
            if len(prefix) >= length:
                if partial:
                    results.append(Trace(events=prefix, meta_data={"freq":prob}))
            else:
                [proba] = model.predict_proba(prefix)
                for i, p in enumerate(proba):
                    if p >= action_prob and prob * p >= path_prob:
                        action = model.classes_[i]
                        if action == TRACE_END:
                            results.append(Trace(events=prefix, meta_data={"freq":prob * p}))
                        else:
                            if self.verbose:
                                print(indent + f" trying {action}")
                            depth_first_search(prefix + [event_factory(action)], prob * p)
        depth_first_search([], 1.0)
        return results

    def execute_test(self, trace:Trace, max_retry:int=0):
        """Executes the given test trace and adds the resulting trace to this set.
        
        Note that if the given trace contains events with missing input values, then
        suitable input values will be generated using 'choose_input_value'.
        Progress messages will be printed if self.verbose is True.
        
        Args:
            trace: the trace to execute (with or without input values).
            max_retry: retry failed operations up to this number of times,
                choosing different random input values each time.
        """
        self.generate_trace(start=True, length=0)
        for ev in trace:
            for i in range(1 + max_retry):
                ev2 = self.call_method(ev.action, ev.inputs, meta_data=ev.meta_data)
                if ev2.status == 0:
                    break
                else:
                    if self.verbose and i < max_retry:
                        print(f"    retry {i+1} ...")


if __name__ == "__main__":
    unittest.main()

