import random
import unittest
from pathlib import Path
import numpy as np
import pytest
import agilkia


THIS_DIR = Path(__file__).parent


class TestObjectiveFunctions(unittest.TestCase):
    event1 = agilkia.Event("Order", {"Name": "Mark"}, {"Status": 0})
    event1b = agilkia.Event("Order", {"Name": "Mark"}, {"Status": 2})
    event2 = agilkia.Event("Skip", {"Size": 3, "Name": "Sue"}, {"Status": 1, "Error": "Too big"})
    trace1 = agilkia.Trace([event1, event1b], meta_data={"freq": 0.6})
    trace2 = agilkia.Trace([event2, event1], meta_data={"freq": 0.5})
    trace3 = agilkia.Trace([event2, event1b], meta_data={"freq": 0.7})
    trace4 = agilkia.Trace([event1, event2], meta_data={"freq": 0.8})
    trace_set = agilkia.TraceSet([trace1, trace2, trace3])
    objective_function = agilkia.ObjectiveFunction()
    frequency_function = agilkia.FrequencyCoverage()

    def test_objective_function(self):
        solution = np.array([1, 1, 0])
        self.objective_function.set_data(self.trace_set, 2)
        self.assertEqual(0, self.objective_function.evaluate(solution))

    def test_objective_function2(self):
        with pytest.raises(ValueError):
            agilkia.ObjectiveFunction(weight=-0.5)

    def test_objective_function3(self):
        with pytest.raises(ValueError):
            self.objective_function.set_data([self.trace1, self.trace2], 1)

    def test_objective_function4(self):
        with pytest.raises(ValueError):
            self.objective_function.set_data(self.trace_set, 0)

    def test_objective_function5(self):
        with pytest.raises(ValueError):
            self.objective_function.set_data(self.trace_set, 4)

    def test_objective_function6(self):
        with pytest.raises(ValueError):
            self.objective_function.set_data(self.trace_set, 2.5)

    def test_frequency_objective_function(self):
        solution = np.array([1, 1, 0])
        self.frequency_function.set_data(self.trace_set, 2)
        self.assertEqual((0.6 + 0.5) / (0.6 + 0.5 + 0.7), self.frequency_function.evaluate(solution))

    def test_frequency_objective_function2(self):
        solution = np.array([1, 1, 1])
        self.frequency_function.set_data(self.trace_set, 2)
        self.assertEqual((np.sum(solution) - 2) * -1, self.frequency_function.evaluate(solution))

    def test_frequency_objective_function3(self):
        trace1 = agilkia.Trace([self.event1, self.event1b])
        trace2 = agilkia.Trace([self.event2, self.event1])
        trace3 = agilkia.Trace([self.event2, self.event1b])
        trace_set = agilkia.TraceSet([trace1, trace2, trace3])
        with pytest.raises(ValueError):
            self.frequency_function.set_data(trace_set, 2)

    def test_action_status_objective_function(self):
        solution = np.array([1, 0, 0])
        objective_function = agilkia.EventCoverage(event_to_str=lambda ev: ev.action + "_" + str(ev.status))
        objective_function.set_data(self.trace_set, 1)
        self.assertEqual(len({"Order_0", "Order_2"}) / len({"Order_0", "Order_2", "Skip_1"}),
                         objective_function.evaluate(solution))

    def test_action_status_objective_function2(self):
        solution = np.array([1, 1, 1])
        trace_set = agilkia.TraceSet([self.trace1, self.trace2, self.trace3])
        objective_function = agilkia.EventCoverage(event_to_str=lambda ev: ev.action + "_" + str(ev.status))
        objective_function.set_data(trace_set, 2)
        self.assertEqual((np.sum(solution) - 2) * -1, objective_function.evaluate(solution))

    def test_action_objective_function(self):
        solution = np.array([1, 0, 0])
        objective_function = agilkia.EventCoverage(event_to_str=lambda ev: ev.action)
        objective_function.set_data(self.trace_set, 1)
        self.assertEqual(len({"Order", "Order"}) / len({"Order", "Order", "Skip"}),
                         objective_function.evaluate(solution))

    def test_action_objective_function2(self):
        solution = np.array([1, 1, 1])
        objective_function = agilkia.EventCoverage(event_to_str=lambda ev: ev.action)
        objective_function.set_data(self.trace_set, 2)
        self.assertEqual((np.sum(solution) - 2) * -1, objective_function.evaluate(solution))

    def test_action_objective_function3(self):
        solution = np.array([1, 1, 1])
        objective_function = agilkia.EventCoverage()
        objective_function.set_data(self.trace_set, 2)
        self.assertEqual((np.sum(solution) - 2) * -1, objective_function.evaluate(solution))

    def test_status_objective_function(self):
        solution = np.array([1, 0, 0])
        objective_function = agilkia.EventCoverage(event_to_str=lambda ev: str(ev.status))
        objective_function.set_data(self.trace_set, 1)
        self.assertEqual(len({"0", "2"}) / len({"0", "2", "1"}), objective_function.evaluate(solution))

    def test_status_objective_function2(self):
        solution = np.array([1, 1, 1])
        objective_function = agilkia.EventCoverage(event_to_str=lambda ev: str(ev.status))
        objective_function.set_data(self.trace_set, 2)
        self.assertEqual((np.sum(solution) - 2) * -1, objective_function.evaluate(solution))

    def test_action_pair_objective_function(self):
        solution = np.array([1, 1, 0, 0])
        trace_set = agilkia.TraceSet([self.trace1, self.trace2, self.trace3, self.trace4])
        objective_function = agilkia.EventPairCoverage(event_to_str=lambda ev: ev.action)
        objective_function.set_data(trace_set, 2)
        self.assertEqual(
            len({"Order_Order", "Skip_Order"}) / len({"Order_Order", "Skip_Order", "Skip_Order", "Order_Skip"}),
            objective_function.evaluate(solution))

    def test_action_pair_objective_function2(self):
        solution = np.array([1, 1, 0, 0])
        trace_set = agilkia.TraceSet([self.trace1, self.trace2, self.trace3, self.trace4])
        objective_function = agilkia.EventPairCoverage()
        objective_function.set_data(trace_set, 2)
        self.assertEqual(
            len({"Order_Order", "Skip_Order"}) / len({"Order_Order", "Skip_Order", "Skip_Order", "Order_Skip"}),
            objective_function.evaluate(solution))

    def test_action_pair_objective_function3(self):
        solution = np.array([1, 1, 1])
        objective_function = agilkia.EventPairCoverage(event_to_str=lambda ev: ev.action)
        objective_function.set_data(self.trace_set, 2)
        self.assertEqual((np.sum(solution) - 2) * -1, objective_function.evaluate(solution))

    def test_status_pair_objective_function(self):
        solution = np.array([1, 1, 0, 0])
        trace_set = agilkia.TraceSet([self.trace1, self.trace2, self.trace3, self.trace4])
        objective_function = agilkia.EventPairCoverage(event_to_str=lambda ev: str(ev.status))
        objective_function.set_data(trace_set, 2)
        self.assertEqual(len({"0_2", "1_0"}) / len({"0_2", "1_0", "1_2", "0_1"}), objective_function.evaluate(solution))

    def test_status_pair_objective_function2(self):
        solution = np.array([1, 1, 1])
        objective_function = agilkia.EventPairCoverage(event_to_str=lambda ev: str(ev.status))
        objective_function.set_data(self.trace_set, 2)
        self.assertEqual((np.sum(solution) - 2) * -1, objective_function.evaluate(solution))

    def test_action_status_pair_objective_function(self):
        solution = np.array([1, 1, 0, 0])
        trace_set = agilkia.TraceSet([self.trace1, self.trace2, self.trace3, self.trace4])
        objective_function = agilkia.EventPairCoverage(event_to_str=lambda ev: ev.action + "_" + str(ev.status))
        objective_function.set_data(trace_set, 2)
        self.assertEqual(len({"Order_0_Order_2", "Skip_1_Order_0"}) / len(
            {"Order_0_Order_2", "Skip_1_Order_0", "Skip_1_Order_2", "Order_0_Skip_1"}),
                         objective_function.evaluate(solution))

    def test_action_status_pair_objective_function2(self):
        solution = np.array([1, 1, 1])
        objective_function = agilkia.EventPairCoverage(event_to_str=lambda ev: ev.action + "_" + str(ev.status))
        objective_function.set_data(self.trace_set, 2)
        self.assertEqual((np.sum(solution) - 2) * -1, objective_function.evaluate(solution))


def set_seed():
    random.seed(3)
    np.random.seed(3)


class TestTraceSetOptimizer(unittest.TestCase):
    event1 = agilkia.Event("Order", {"Name": "Mark"}, {"Status": 0})
    event1b = agilkia.Event("Order", {"Name": "Mark"}, {"Status": 2})
    event2 = agilkia.Event("Skip", {"Size": 3, "Name": "Sue"}, {"Status": 1, "Error": "Too big"})
    trace1 = agilkia.Trace([event1, event1b], meta_data={"freq": 0.6})
    trace2 = agilkia.Trace([event2, event1], meta_data={"freq": 0.5})
    trace3 = agilkia.Trace([event2, event1b], meta_data={"freq": 0.7})
    trace4 = agilkia.Trace([event1, event2], meta_data={"freq": 0.8})
    trace_set = agilkia.TraceSet([trace1, trace2, trace3, trace4])

    def test_trace_set_optimizer(self):
        objective_function = agilkia.FrequencyCoverage()
        agilkia.TraceSetOptimizer(objective_function)

    def test_trace_set_optimizer2(self):
        with pytest.raises(ValueError):
            agilkia.TraceSetOptimizer(lambda ev: ev.action)

    def test_trace_set_optimizer3(self):
        with pytest.raises(ValueError):
            objective_function = agilkia.FrequencyCoverage()
            traceset_optimizer = agilkia.TraceSetOptimizer(objective_function)
            traceset_optimizer.set_data([self.trace1, self.trace2], 1)

    def test_trace_set_optimizer4(self):
        with pytest.raises(ValueError):
            objective_function = agilkia.FrequencyCoverage()
            traceset_optimizer = agilkia.TraceSetOptimizer(objective_function)
            traceset_optimizer.set_data(self.trace_set, 0)

    def test_trace_set_optimizer5(self):
        with pytest.raises(ValueError):
            objective_function = agilkia.FrequencyCoverage()
            traceset_optimizer = agilkia.TraceSetOptimizer(objective_function)
            traceset_optimizer.set_data(self.trace_set, 5)

    def test_trace_set_optimizer6(self):
        with pytest.raises(ValueError):
            objective_function = agilkia.FrequencyCoverage()
            traceset_optimizer = agilkia.TraceSetOptimizer(objective_function)
            traceset_optimizer.set_data(self.trace_set, 2.5)

    def test_greedy(self):
        objective_functions = [agilkia.FrequencyCoverage(), agilkia.EventCoverage(event_to_str=lambda ev: ev.action)]
        greedy_optimizer = agilkia.GreedyOptimizer(objective_functions)
        greedy_optimizer.set_data(self.trace_set, 1)
        selected_traces, best_objective_value = greedy_optimizer.optimize()
        self.assertEqual([self.trace4], selected_traces.traces)
        self.assertEqual(((3 / 3) * 0.5 + (0.8 / 2.6) * 0.5), best_objective_value)

    def test_greedy2(self):
        objective_functions = [agilkia.FrequencyCoverage(),
                               agilkia.EventCoverage(event_to_str=lambda ev: ev.action + "_" + str(ev.status))]
        greedy_optimizer = agilkia.GreedyOptimizer(objective_functions)
        greedy_optimizer.set_data(self.trace_set, 1)
        selected_traces, best_objective_value = greedy_optimizer.optimize()
        self.assertEqual([self.trace4], selected_traces.traces)
        self.assertEqual(((2 / 3) * 0.5 + (0.8 / 2.6) * 0.5), best_objective_value)

    def test_greedy3(self):
        objective_functions = [agilkia.FrequencyCoverage(),
                               agilkia.EventCoverage(event_to_str=lambda ev: str(ev.status))]
        greedy_optimizer = agilkia.GreedyOptimizer(objective_functions)
        greedy_optimizer.set_data(self.trace_set, 1)
        selected_traces, best_objective_value = greedy_optimizer.optimize()
        self.assertEqual([self.trace4], selected_traces.traces)
        self.assertEqual(((2 / 3) * 0.5 + (0.8 / 2.6) * 0.5), best_objective_value)

    def test_greedy4(self):
        objective_functions = [agilkia.FrequencyCoverage(),
                               agilkia.EventPairCoverage(event_to_str=lambda ev: ev.action)]
        greedy_optimizer = agilkia.GreedyOptimizer(objective_functions)
        greedy_optimizer.set_data(self.trace_set, 2)
        selected_traces, best_objective_value = greedy_optimizer.optimize()
        self.assertEqual({self.trace4, self.trace3}, set(selected_traces.traces))
        self.assertEqual(((2 / 3) * 0.5 + (1.5 / 2.6) * 0.5), best_objective_value)

    def test_greedy5(self):
        objective_functions = [agilkia.FrequencyCoverage(),
                               agilkia.EventPairCoverage(event_to_str=lambda ev: ev.action + "_" + str(ev.status))]
        greedy_optimizer = agilkia.GreedyOptimizer(objective_functions)
        greedy_optimizer.set_data(self.trace_set, 2)
        selected_traces, best_objective_value = greedy_optimizer.optimize()
        self.assertEqual({self.trace4, self.trace3}, set(selected_traces.traces))
        self.assertEqual(((2 / 4) * 0.5 + (1.5 / 2.6) * 0.5), best_objective_value)

    def test_greedy6(self):
        objective_functions = [agilkia.FrequencyCoverage(),
                               agilkia.EventPairCoverage(event_to_str=lambda ev: str(ev.status))]
        greedy_optimizer = agilkia.GreedyOptimizer(objective_functions)
        greedy_optimizer.set_data(self.trace_set, 2)
        selected_traces, best_objective_value = greedy_optimizer.optimize()
        self.assertEqual({self.trace4, self.trace3}, set(selected_traces.traces))
        self.assertEqual(((2 / 4) * 0.5 + (1.5 / 2.6) * 0.5), best_objective_value)

    def test_pso(self):
        objective_functions = [agilkia.FrequencyCoverage(), agilkia.EventCoverage(event_to_str=lambda ev: ev.action)]
        pso_optimizer = agilkia.ParticleSwarmOptimizer(objective_functions)
        pso_optimizer.set_data(self.trace_set, select=1)
        selected_traces, best_objective_value = pso_optimizer.optimize()
        self.assertEqual([self.trace4], selected_traces.traces)
        self.assertEqual(((3 / 3) * 0.5 + (0.8 / 2.6) * 0.5), best_objective_value)

    def test_pso2(self):
        objective_functions = [agilkia.FrequencyCoverage(),
                               agilkia.EventCoverage(event_to_str=lambda ev: ev.action + "_" + str(ev.status))]
        pso_optimizer = agilkia.ParticleSwarmOptimizer(objective_functions)
        pso_optimizer.set_data(self.trace_set, select=1)
        selected_traces, best_objective_value = pso_optimizer.optimize()
        self.assertEqual([self.trace4], selected_traces.traces)
        self.assertEqual(((2 / 3) * 0.5 + (0.8 / 2.6) * 0.5), best_objective_value)

    def test_pso3(self):
        objective_functions = [agilkia.FrequencyCoverage(),
                               agilkia.EventCoverage(event_to_str=lambda ev: str(ev.status))]
        pso_optimizer = agilkia.ParticleSwarmOptimizer(objective_functions)
        pso_optimizer.set_data(self.trace_set, select=1)
        selected_traces, best_objective_value = pso_optimizer.optimize()
        self.assertEqual([self.trace4], selected_traces.traces)
        self.assertEqual(((2 / 3) * 0.5 + (0.8 / 2.6) * 0.5), best_objective_value)

    def test_pso4(self):
        objective_functions = [agilkia.FrequencyCoverage(),
                               agilkia.EventPairCoverage(event_to_str=lambda ev: ev.action)]
        pso_optimizer = agilkia.ParticleSwarmOptimizer(objective_functions)
        pso_optimizer.set_data(self.trace_set, select=2)
        selected_traces, best_objective_value = pso_optimizer.optimize()
        self.assertEqual({self.trace4, self.trace3}, set(selected_traces.traces))
        self.assertEqual(((2 / 3) * 0.5 + (1.5 / 2.6) * 0.5), best_objective_value)

    def test_pso5(self):
        objective_functions = [agilkia.FrequencyCoverage(),
                               agilkia.EventPairCoverage(event_to_str=lambda ev: ev.action + "_" + str(ev.status))]
        pso_optimizer = agilkia.ParticleSwarmOptimizer(objective_functions)
        pso_optimizer.set_data(self.trace_set, select=2)
        selected_traces, best_objective_value = pso_optimizer.optimize()
        self.assertEqual({self.trace4, self.trace3}, set(selected_traces.traces))
        self.assertEqual(((2 / 4) * 0.5 + (1.5 / 2.6) * 0.5), best_objective_value)

    def test_pso6(self):
        objective_functions = [agilkia.FrequencyCoverage(),
                               agilkia.EventPairCoverage(event_to_str=lambda ev: str(ev.status))]
        pso_optimizer = agilkia.ParticleSwarmOptimizer(objective_functions)
        pso_optimizer.set_data(self.trace_set, select=2)
        selected_traces, best_objective_value = pso_optimizer.optimize()
        self.assertEqual({self.trace4, self.trace3}, set(selected_traces.traces))
        self.assertEqual(((2 / 4) * 0.5 + (1.5 / 2.6) * 0.5), best_objective_value)

    def test_pso7(self):
        objective_functions = [agilkia.FrequencyCoverage(),
                               agilkia.EventPairCoverage(event_to_str=lambda ev: str(ev.status))]
        with pytest.raises(ValueError):
            agilkia.ParticleSwarmOptimizer(objective_functions, num_of_particles=0)

    def test_pso8(self):
        objective_functions = [agilkia.FrequencyCoverage(),
                               agilkia.EventPairCoverage(event_to_str=lambda ev: str(ev.status))]
        with pytest.raises(ValueError):
            agilkia.ParticleSwarmOptimizer(objective_functions, num_of_iterations=0)

    def test_pso9(self):
        objective_functions = [agilkia.FrequencyCoverage(),
                               agilkia.EventPairCoverage(event_to_str=lambda ev: str(ev.status))]
        with pytest.raises(ValueError):
            agilkia.ParticleSwarmOptimizer(objective_functions, c1=0)

    def test_pso10(self):
        objective_functions = [agilkia.FrequencyCoverage(),
                               agilkia.EventPairCoverage(event_to_str=lambda ev: str(ev.status))]
        with pytest.raises(ValueError):
            agilkia.ParticleSwarmOptimizer(objective_functions, c2=0)

    def test_ga(self):
        set_seed()
        objective_functions = [agilkia.FrequencyCoverage(),
                               agilkia.EventPairCoverage(event_to_str=lambda ev: ev.action)]
        ga_optimizer = agilkia.GeneticOptimizer(objective_functions)
        ga_optimizer.set_data(self.trace_set, select=2)
        selected_traces, best_objective_value = ga_optimizer.optimize()
        self.assertEqual({self.trace4, self.trace3}, set(selected_traces.traces))
        self.assertEqual(((2 / 3) * 0.5 + (1.5 / 2.6) * 0.5), best_objective_value)

    def test_ga2(self):
        set_seed()
        objective_functions = [agilkia.FrequencyCoverage(),
                               agilkia.EventCoverage(event_to_str=lambda ev: ev.action + "_" + str(ev.status))]
        ga_optimizer = agilkia.GeneticOptimizer(objective_functions)
        ga_optimizer.set_data(self.trace_set, select=1)
        selected_traces, best_objective_value = ga_optimizer.optimize()
        self.assertEqual([self.trace4], selected_traces.traces)
        self.assertEqual(((2 / 3) * 0.5 + (0.8 / 2.6) * 0.5), best_objective_value)

    def test_ga3(self):
        set_seed()
        objective_functions = [agilkia.FrequencyCoverage(),
                               agilkia.EventCoverage(event_to_str=lambda ev: str(ev.status))]
        ga_optimizer = agilkia.GeneticOptimizer(objective_functions)
        ga_optimizer.set_data(self.trace_set, select=1)
        selected_traces, best_objective_value = ga_optimizer.optimize()
        self.assertEqual([self.trace4], selected_traces.traces)
        self.assertEqual(((2 / 3) * 0.5 + (0.8 / 2.6) * 0.5), best_objective_value)

    def test_ga4(self):
        set_seed()
        objective_functions = [agilkia.FrequencyCoverage(),
                               agilkia.EventPairCoverage(event_to_str=lambda ev: ev.action)]
        ga_optimizer = agilkia.GeneticOptimizer(objective_functions)
        ga_optimizer.set_data(self.trace_set, select=2)
        selected_traces, best_objective_value = ga_optimizer.optimize()
        self.assertEqual({self.trace4, self.trace3}, set(selected_traces.traces))
        self.assertEqual(((2 / 3) * 0.5 + (1.5 / 2.6) * 0.5), best_objective_value)

    def test_ga5(self):
        set_seed()
        objective_functions = [agilkia.FrequencyCoverage(),
                               agilkia.EventPairCoverage(event_to_str=lambda ev: ev.action + "_" + str(ev.status))]
        ga_optimizer = agilkia.GeneticOptimizer(objective_functions)
        ga_optimizer.set_data(self.trace_set, select=2)
        selected_traces, best_objective_value = ga_optimizer.optimize()
        self.assertEqual({self.trace4, self.trace3}, set(selected_traces.traces))
        self.assertEqual(((2 / 4) * 0.5 + (1.5 / 2.6) * 0.5), best_objective_value)

    def test_ga6(self):
        set_seed()
        objective_functions = [agilkia.FrequencyCoverage(),
                               agilkia.EventPairCoverage(event_to_str=lambda ev: str(ev.status))]
        ga_optimizer = agilkia.GeneticOptimizer(objective_functions)
        ga_optimizer.set_data(self.trace_set, select=2)
        selected_traces, best_objective_value = ga_optimizer.optimize()
        self.assertEqual({self.trace4, self.trace3}, set(selected_traces.traces))
        self.assertEqual(((2 / 4) * 0.5 + (1.5 / 2.6) * 0.5), best_objective_value)

    def test_ga7(self):
        set_seed()
        objective_functions = [agilkia.FrequencyCoverage(),
                               agilkia.EventCoverage(event_to_str=lambda ev: str(ev.status))]
        ga_optimizer = agilkia.GeneticOptimizer(objective_functions, crossover="single")
        ga_optimizer.set_data(self.trace_set, select=1)
        selected_traces, best_objective_value = ga_optimizer.optimize()
        self.assertEqual([self.trace4], selected_traces.traces)
        self.assertEqual(((2 / 3) * 0.5 + (0.8 / 2.6) * 0.5), best_objective_value)

    def test_ga8(self):
        set_seed()
        objective_functions = [agilkia.FrequencyCoverage(),
                               agilkia.EventCoverage(event_to_str=lambda ev: str(ev.status))]
        with pytest.raises(ValueError):
            agilkia.GeneticOptimizer(objective_functions, crossover="triple")

    def test_ga9(self):
        set_seed()
        objective_functions = [agilkia.FrequencyCoverage(),
                               agilkia.EventCoverage(event_to_str=lambda ev: str(ev.status))]
        with pytest.raises(ValueError):
            agilkia.GeneticOptimizer(objective_functions, num_of_iterations=0)

    def test_ga10(self):
        set_seed()
        objective_functions = [agilkia.FrequencyCoverage(),
                               agilkia.EventCoverage(event_to_str=lambda ev: str(ev.status))]
        with pytest.raises(ValueError):
            agilkia.GeneticOptimizer(objective_functions, num_of_chromosomes=0)

    def test_ga11(self):
        set_seed()
        objective_functions = [agilkia.FrequencyCoverage(),
                               agilkia.EventCoverage(event_to_str=lambda ev: str(ev.status))]
        with pytest.raises(ValueError):
            agilkia.GeneticOptimizer(objective_functions, prob_cross=-0.1)

    def test_ga12(self):
        set_seed()
        objective_functions = [agilkia.FrequencyCoverage(),
                               agilkia.EventCoverage(event_to_str=lambda ev: str(ev.status))]
        with pytest.raises(ValueError):
            agilkia.GeneticOptimizer(objective_functions, prob_mutate=-0.1)

    def test_ga13(self):
        set_seed()
        objective_functions = [agilkia.FrequencyCoverage(),
                               agilkia.EventCoverage(event_to_str=lambda ev: str(ev.status))]
        with pytest.raises(ValueError):
            agilkia.GeneticOptimizer(objective_functions, prob_cross=2)

    def test_ga14(self):
        set_seed()
        objective_functions = [agilkia.FrequencyCoverage(),
                               agilkia.EventCoverage(event_to_str=lambda ev: str(ev.status))]
        with pytest.raises(ValueError):
            agilkia.GeneticOptimizer(objective_functions, prob_cross=2.5)

    def test_ga15(self):
        set_seed()
        objective_functions = [agilkia.FrequencyCoverage(),
                               agilkia.EventCoverage(event_to_str=lambda ev: str(ev.status))]
        with pytest.raises(ValueError):
            agilkia.GeneticOptimizer(objective_functions, prob_mutate=2)

    def test_ga16(self):
        set_seed()
        objective_functions = [agilkia.FrequencyCoverage(),
                               agilkia.EventCoverage(event_to_str=lambda ev: str(ev.status))]
        with pytest.raises(ValueError):
            agilkia.GeneticOptimizer(objective_functions, prob_mutate=2.5)


class TestScanner(unittest.TestCase):
    trace_set = agilkia.TraceSet.load_from_json(Path(THIS_DIR / "fixtures" / "scanner.json"))
    objective_functions = [agilkia.FrequencyCoverage(),
                           agilkia.EventCoverage(event_to_str=lambda ev: ev.action + "_" + str(ev.status))]
    greedy_optimizer = agilkia.GreedyOptimizer(objective_functions)
    pso_optimizer = agilkia.ParticleSwarmOptimizer(objective_functions)
    ga_optimizer = agilkia.GeneticOptimizer(objective_functions)
    brute_force_results = [0.363, 0.455, 0.494, 0.532, 0.569, 0.606, 0.642, 0.679, 0.714]

    def test_greedy(self):
        for i in range(2, 11):
            self.greedy_optimizer.set_data(self.trace_set, select=i)
            selected_traces, best_objective_value = self.greedy_optimizer.optimize()
            assert best_objective_value == pytest.approx(self.brute_force_results[i - 2], 0.01)

    def test_pso(self):
        for i in range(2, 11):
            self.pso_optimizer.set_data(self.trace_set, select=i)
            selected_traces, best_objective_value = self.pso_optimizer.optimize()
            assert best_objective_value == pytest.approx(self.brute_force_results[i - 2], 0.01)

    def test_ga(self):
        for i in range(2, 11):
            set_seed()
            self.ga_optimizer.set_data(self.trace_set, select=i)
            selected_traces, best_objective_value = self.ga_optimizer.optimize()
            assert best_objective_value == pytest.approx(self.brute_force_results[i - 2], 0.01)
