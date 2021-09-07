import unittest
import numpy as np
import pytest

import agilkia


class TestObjectiveFunctions(unittest.TestCase):
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
        self.assertEqual((0.6 + 0.5) / (0.6 + 0.5 + 0.7), objective_function.evaluate(solution))

    def test_frequency_objective_function2(self):
        solution = np.array([1, 1, 1])
        objective_function = agilkia.FrequencyCoverage(self.trace_set, 2)
        self.assertEqual((np.sum(solution) - 2) * -1, objective_function.evaluate(solution))

    def test_frequency_objective_function3(self):
        trace1 = agilkia.Trace([self.event1, self.event1b])
        trace2 = agilkia.Trace([self.event2, self.event1])
        trace3 = agilkia.Trace([self.event2, self.event1b])
        trace_set = agilkia.TraceSet([trace1, trace2, trace3])
        with pytest.raises(ValueError):
            agilkia.FrequencyCoverage(trace_set, 2)

    def test_action_status_objective_function(self):
        solution = np.array([1, 0, 0])
        objective_function = agilkia.EventCoverage(self.trace_set, 1,
                                                   event_to_str=lambda ev: ev.action + "_" + str(ev.status))
        self.assertEqual(len({"Order_0", "Order_2"}) / len({"Order_0", "Order_2", "Skip_1"}),
                         objective_function.evaluate(solution))

    def test_action_status_objective_function2(self):
        solution = np.array([1, 1, 1])
        trace_set = agilkia.TraceSet([self.trace1, self.trace2, self.trace3])
        objective_function = agilkia.EventCoverage(trace_set, 2,
                                                   event_to_str=lambda ev: ev.action + "_" + str(ev.status))
        self.assertEqual((np.sum(solution) - 2) * -1, objective_function.evaluate(solution))

    def test_action_objective_function(self):
        solution = np.array([1, 0, 0])
        objective_function = agilkia.EventCoverage(self.trace_set, 1, event_to_str=lambda ev: ev.action)
        self.assertEqual(len({"Order", "Order"}) / len({"Order", "Order", "Skip"}),
                         objective_function.evaluate(solution))

    def test_action_objective_function2(self):
        solution = np.array([1, 1, 1])
        objective_function = agilkia.EventCoverage(self.trace_set, 2, event_to_str=lambda ev: ev.action)
        self.assertEqual((np.sum(solution) - 2) * -1, objective_function.evaluate(solution))

    def test_status_objective_function(self):
        solution = np.array([1, 0, 0])
        objective_function = agilkia.EventCoverage(self.trace_set, 1, event_to_str=lambda ev: str(ev.status))
        self.assertEqual(len({"0", "2"}) / len({"0", "2", "1"}), objective_function.evaluate(solution))

    def test_status_objective_function2(self):
        solution = np.array([1, 1, 1])
        objective_function = agilkia.EventCoverage(self.trace_set, 2, event_to_str=lambda ev: str(ev.status))
        self.assertEqual((np.sum(solution) - 2) * -1, objective_function.evaluate(solution))

    def test_action_pair_objective_function(self):
        solution = np.array([1, 1, 0, 0])
        trace_set = agilkia.TraceSet([self.trace1, self.trace2, self.trace3, self.trace4])
        objective_function = agilkia.EventPairCoverage(trace_set, 2, event_to_str=lambda ev: ev.action)
        self.assertEqual(
            len({"Order_Order", "Skip_Order"}) / len({"Order_Order", "Skip_Order", "Skip_Order", "Order_Skip"}),
            objective_function.evaluate(solution))

    def test_action_pair_objective_function2(self):
        solution = np.array([1, 1, 1])
        objective_function = agilkia.EventPairCoverage(self.trace_set, 2, event_to_str=lambda ev: ev.action)
        self.assertEqual((np.sum(solution) - 2) * -1, objective_function.evaluate(solution))

    def test_status_pair_objective_function(self):
        solution = np.array([1, 1, 0, 0])
        trace_set = agilkia.TraceSet([self.trace1, self.trace2, self.trace3, self.trace4])
        objective_function = agilkia.EventPairCoverage(trace_set, 2, event_to_str=lambda ev: str(ev.status))
        self.assertEqual(len({"0_2", "1_0"}) / len({"0_2", "1_0", "1_2", "0_1"}), objective_function.evaluate(solution))

    def test_status_pair_objective_function2(self):
        solution = np.array([1, 1, 1])
        objective_function = agilkia.EventPairCoverage(self.trace_set, 2, event_to_str=lambda ev: str(ev.status))
        self.assertEqual((np.sum(solution) - 2) * -1, objective_function.evaluate(solution))

    def test_action_status_pair_objective_function(self):
        solution = np.array([1, 1, 0, 0])
        trace_set = agilkia.TraceSet([self.trace1, self.trace2, self.trace3, self.trace4])
        objective_function = agilkia.EventPairCoverage(trace_set, 2,
                                                       event_to_str=lambda ev: ev.action + "_" + str(ev.status))
        self.assertEqual(len({"Order_0_Order_2", "Skip_1_Order_0"}) / len(
            {"Order_0_Order_2", "Skip_1_Order_0", "Skip_1_Order_2", "Order_0_Skip_1"}),
                         objective_function.evaluate(solution))
