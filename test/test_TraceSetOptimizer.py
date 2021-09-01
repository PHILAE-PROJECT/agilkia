import unittest
import numpy as np
import agilkia


class TestTraceSetOptimizer(unittest.TestCase):
    event1 = agilkia.Event("Order", {"Name": "Mark"}, {"Status": 0})
    event1b = agilkia.Event("Order", {"Name": "Mark"}, {"Status": 2})
    event2 = agilkia.Event("Skip", {"Size": 3, "Name": "Sue"}, {"Status": 1, "Error": "Too big"})
    trace1 = agilkia.Trace([event1, event1b], meta_data={"freq": 0.6})
    trace2 = agilkia.Trace([event2, event1], meta_data={"freq": 0.5})
    trace3 = agilkia.Trace([event2, event1b], meta_data={"freq": 0.7})
    trace4 = agilkia.Trace([event1, event2], meta_data={"freq": 0.8})
    trace_set = agilkia.TraceSet([trace1, trace2, trace3])

    def test_objective_function(self):
        solution = np.array([1, 1, 0])
        objective_function = agilkia.ObjectiveFunction(self.trace_set, 2)
        self.assertEqual(0, objective_function.evaluate(solution))

    def test_frequency_objective_function(self):
        solution = np.array([1, 1, 0])
        objective_function = agilkia.FrequencyCoverage(self.trace_set, 2)
        self.assertEqual((self.trace1.get_meta('freq', 0) + self.trace2.get_meta('freq', 0)) / (
                self.trace1.get_meta('freq', 0) + self.trace2.get_meta('freq', 0) + self.trace3.get_meta('freq', 0)),
                         objective_function.evaluate(solution))

    def test_frequency_objective_function2(self):
        solution = np.array([1, 1, 1])
        objective_function = agilkia.FrequencyCoverage(self.trace_set, 2)
        self.assertEqual((np.sum(solution) - 2) * -1, objective_function.evaluate(solution))

    def test_action_status_objective_function(self):
        solution = np.array([1, 0, 0])
        objective_function = agilkia.ActionStatusCoverage(self.trace_set, 1)
        self.assertEqual(len({self.event1.action + "_" + str(self.event1.status),
                              self.event1b.action + "_" + str(self.event1b.status)}) / len(
            {self.event1.action + "_" + str(self.event1.status), self.event1b.action + "_" + str(self.event1b.status),
             self.event2.action + "_" + str(self.event2.status)}), objective_function.evaluate(solution))

    def test_action_status_objective_function2(self):
        solution = np.array([1, 1, 1])
        trace_set = agilkia.TraceSet([self.trace1, self.trace2, self.trace3])
        objective_function = agilkia.ActionStatusCoverage(trace_set, 2)
        self.assertEqual((np.sum(solution) - 2) * -1, objective_function.evaluate(solution))

    def test_action_objective_function(self):
        solution = np.array([1, 0, 0])
        objective_function = agilkia.ActionCoverage(self.trace_set, 1)
        self.assertEqual(len({self.event1.action, self.event1b.action}) / len(
            {self.event1.action, self.event1b.action, self.event2.action}), objective_function.evaluate(solution))

    def test_action_objective_function2(self):
        solution = np.array([1, 1, 1])
        objective_function = agilkia.ActionCoverage(self.trace_set, 2)
        self.assertEqual((np.sum(solution) - 2) * -1, objective_function.evaluate(solution))

    def test_status_objective_function(self):
        solution = np.array([1, 0, 0])
        objective_function = agilkia.StatusCoverage(self.trace_set, 1)
        self.assertEqual(len({self.event1.status, self.event1b.status}) / len(
            {self.event1.status, self.event1b.status, self.event2.status}), objective_function.evaluate(solution))

    def test_status_objective_function2(self):
        solution = np.array([1, 1, 1])
        objective_function = agilkia.StatusCoverage(self.trace_set, 2)
        self.assertEqual((np.sum(solution) - 2) * -1, objective_function.evaluate(solution))

    def test_action_pair_objective_function(self):
        solution = np.array([1, 1, 0, 0])
        trace_set = agilkia.TraceSet([self.trace1, self.trace2, self.trace3, self.trace4])
        objective_function = agilkia.ActionPairCoverage(trace_set, 2)
        self.assertEqual(
            len({self.event1.action + "_" + self.event1b.action, self.event2.action + "_" + self.event1.action}) / len(
                {self.event1.action + "_" + self.event1b.action, self.event2.action + "_" + self.event1.action,
                 self.event2.action + "_" + self.event1b.action, self.event1.action + "_" + self.event2.action}),
            objective_function.evaluate(solution))

    def test_action_pair_objective_function2(self):
        solution = np.array([1, 1, 1])
        objective_function = agilkia.ActionPairCoverage(self.trace_set, 2)
        self.assertEqual((np.sum(solution) - 2) * -1, objective_function.evaluate(solution))