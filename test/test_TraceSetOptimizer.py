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

    def test_action_status_pair_objective_function2(self):
        solution = np.array([1, 1, 1])
        objective_function = agilkia.EventPairCoverage(self.trace_set, 2,
                                                       event_to_str=lambda ev: ev.action + "_" + str(ev.status))
        self.assertEqual((np.sum(solution) - 2) * -1, objective_function.evaluate(solution))


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
        objective_function = agilkia.FrequencyCoverage(self.trace_set, 1)
        agilkia.TraceSetOptimizer(self.trace_set, objective_function, 1)

    def test_trace_set_optimizer2(self):
        with pytest.raises(ValueError):
            agilkia.TraceSetOptimizer(self.trace_set, lambda ev: ev.action, 1)

    def test_greedy(self):
        objective_functions = []
        frequency_function = agilkia.FrequencyCoverage(self.trace_set, 1)
        action_function = agilkia.EventCoverage(self.trace_set, 1, event_to_str=lambda ev: ev.action)
        objective_functions.extend([frequency_function, action_function])
        greedy_optimizer = agilkia.GreedyOptimizer(self.trace_set, objective_functions, 1)
        selected_traces, best_objective_value = greedy_optimizer.optimize()
        self.assertEqual([self.trace4], selected_traces.traces)
        self.assertEqual(((3 / 3) * 0.5 + (0.8 / 2.6) * 0.5), best_objective_value)

    def test_greedy2(self):
        objective_functions = []
        frequency_function = agilkia.FrequencyCoverage(self.trace_set, 1)
        action_status_function = agilkia.EventCoverage(self.trace_set, 1,
                                                       event_to_str=lambda ev: ev.action + "_" + str(ev.status))
        objective_functions.extend([frequency_function, action_status_function])
        greedy_optimizer = agilkia.GreedyOptimizer(self.trace_set, objective_functions, 1)
        selected_traces, best_objective_value = greedy_optimizer.optimize()
        self.assertEqual([self.trace4], selected_traces.traces)
        self.assertEqual(((2 / 3) * 0.5 + (0.8 / 2.6) * 0.5), best_objective_value)

    def test_greedy3(self):
        frequency_function = agilkia.FrequencyCoverage(self.trace_set, 1)
        status_function = agilkia.EventCoverage(self.trace_set, 1, event_to_str=lambda ev: str(ev.status))
        objective_functions = [frequency_function, status_function]
        greedy_optimizer = agilkia.GreedyOptimizer(self.trace_set, objective_functions, 1)
        selected_traces, best_objective_value = greedy_optimizer.optimize()
        self.assertEqual([self.trace4], selected_traces.traces)
        self.assertEqual(((2 / 3) * 0.5 + (0.8 / 2.6) * 0.5), best_objective_value)

    def test_greedy4(self):
        objective_functions = []
        frequency_function = agilkia.FrequencyCoverage(self.trace_set, 2)
        action_pair_function = agilkia.EventPairCoverage(self.trace_set, 2, event_to_str=lambda ev: ev.action)
        objective_functions.extend([frequency_function, action_pair_function])
        greedy_optimizer = agilkia.GreedyOptimizer(self.trace_set, objective_functions, 2)
        selected_traces, best_objective_value = greedy_optimizer.optimize()
        self.assertEqual({self.trace4, self.trace3}, set(selected_traces.traces))
        self.assertEqual(((2 / 3) * 0.5 + (1.5 / 2.6) * 0.5), best_objective_value)

    def test_greedy5(self):
        objective_functions = []
        frequency_function = agilkia.FrequencyCoverage(self.trace_set, 2)
        action_pair_function = agilkia.EventPairCoverage(self.trace_set, 2,
                                                         event_to_str=lambda ev: ev.action + "_" + str(ev.status))
        objective_functions.extend([frequency_function, action_pair_function])
        greedy_optimizer = agilkia.GreedyOptimizer(self.trace_set, objective_functions, 2)
        selected_traces, best_objective_value = greedy_optimizer.optimize()
        self.assertEqual({self.trace4, self.trace3}, set(selected_traces.traces))
        self.assertEqual(((2 / 4) * 0.5 + (1.5 / 2.6) * 0.5), best_objective_value)

    def test_greedy6(self):
        objective_functions = []
        frequency_function = agilkia.FrequencyCoverage(self.trace_set, 2)
        action_pair_function = agilkia.EventPairCoverage(self.trace_set, 2, event_to_str=lambda ev: str(ev.status))
        objective_functions.extend([frequency_function, action_pair_function])
        greedy_optimizer = agilkia.GreedyOptimizer(self.trace_set, objective_functions, 2)
        selected_traces, best_objective_value = greedy_optimizer.optimize()
        self.assertEqual({self.trace4, self.trace3}, set(selected_traces.traces))
        self.assertEqual(((2 / 4) * 0.5 + (1.5 / 2.6) * 0.5), best_objective_value)

    def test_pso(self):
        objective_functions = []
        frequency_function = agilkia.FrequencyCoverage(self.trace_set, 1)
        action_function = agilkia.EventCoverage(self.trace_set, 1, event_to_str=lambda ev: ev.action)
        objective_functions.extend([frequency_function, action_function])
        pso_optimizer = agilkia.ParticleSwarmOptimizer(self.trace_set, objective_functions, num_of_selected_traces=2,
                                                       number_of_particles=50, number_of_iterations=50, c1=2.0,
                                                       c2=2.0)
        selected_traces, best_objective_value = pso_optimizer.optimize()
        self.assertEqual([self.trace4], selected_traces.traces)
        self.assertEqual(((3 / 3) * 0.5 + (0.8 / 2.6) * 0.5), best_objective_value)

    def test_pso2(self):
        objective_functions = []
        frequency_function = agilkia.FrequencyCoverage(self.trace_set, 1)
        action_status_function = agilkia.EventCoverage(self.trace_set, 1,
                                                       event_to_str=lambda ev: ev.action + "_" + str(ev.status))
        objective_functions.extend([frequency_function, action_status_function])
        pso_optimizer = agilkia.ParticleSwarmOptimizer(self.trace_set, objective_functions, num_of_selected_traces=2,
                                                       number_of_particles=50, number_of_iterations=50, c1=2.0,
                                                       c2=2.0)
        selected_traces, best_objective_value = pso_optimizer.optimize()
        self.assertEqual([self.trace4], selected_traces.traces)
        self.assertEqual(((2 / 3) * 0.5 + (0.8 / 2.6) * 0.5), best_objective_value)

    def test_pso3(self):
        frequency_function = agilkia.FrequencyCoverage(self.trace_set, 1)
        status_function = agilkia.EventCoverage(self.trace_set, 1, event_to_str=lambda ev: str(ev.status))
        objective_functions = [frequency_function, status_function]
        pso_optimizer = agilkia.ParticleSwarmOptimizer(self.trace_set, objective_functions, num_of_selected_traces=2,
                                                       number_of_particles=50, number_of_iterations=50, c1=2.0,
                                                       c2=2.0)
        selected_traces, best_objective_value = pso_optimizer.optimize()
        self.assertEqual([self.trace4], selected_traces.traces)
        self.assertEqual(((2 / 3) * 0.5 + (0.8 / 2.6) * 0.5), best_objective_value)

    def test_pso4(self):
        objective_functions = []
        frequency_function = agilkia.FrequencyCoverage(self.trace_set, 2)
        action_pair_function = agilkia.EventPairCoverage(self.trace_set, 2, event_to_str=lambda ev: ev.action)
        objective_functions.extend([frequency_function, action_pair_function])
        pso_optimizer = agilkia.ParticleSwarmOptimizer(self.trace_set, objective_functions, num_of_selected_traces=2,
                                                       number_of_particles=50, number_of_iterations=50, c1=2.0,
                                                       c2=2.0)
        selected_traces, best_objective_value = pso_optimizer.optimize()
        self.assertEqual({self.trace4, self.trace3}, set(selected_traces.traces))
        self.assertEqual(((2 / 3) * 0.5 + (1.5 / 2.6) * 0.5), best_objective_value)

    def test_pso5(self):
        objective_functions = []
        frequency_function = agilkia.FrequencyCoverage(self.trace_set, 2)
        action_pair_function = agilkia.EventPairCoverage(self.trace_set, 2,
                                                         event_to_str=lambda ev: ev.action + "_" + str(ev.status))
        objective_functions.extend([frequency_function, action_pair_function])
        pso_optimizer = agilkia.ParticleSwarmOptimizer(self.trace_set, objective_functions, num_of_selected_traces=2,
                                                       number_of_particles=50, number_of_iterations=50, c1=2.0,
                                                       c2=2.0)
        selected_traces, best_objective_value = pso_optimizer.optimize()
        self.assertEqual({self.trace4, self.trace3}, set(selected_traces.traces))
        self.assertEqual(((2 / 4) * 0.5 + (1.5 / 2.6) * 0.5), best_objective_value)

    def test_pso6(self):
        objective_functions = []
        frequency_function = agilkia.FrequencyCoverage(self.trace_set, 2)
        action_pair_function = agilkia.EventPairCoverage(self.trace_set, 2, event_to_str=lambda ev: str(ev.status))
        objective_functions.extend([frequency_function, action_pair_function])
        pso_optimizer = agilkia.ParticleSwarmOptimizer(self.trace_set, objective_functions, num_of_selected_traces=2,
                                                       number_of_particles=50, number_of_iterations=50, c1=2.0,
                                                       c2=2.0)
        selected_traces, best_objective_value = pso_optimizer.optimize()
        self.assertEqual({self.trace4, self.trace3}, set(selected_traces.traces))
        self.assertEqual(((2 / 4) * 0.5 + (1.5 / 2.6) * 0.5), best_objective_value)

    def test_ga(self):
        objective_functions = []
        frequency_function = agilkia.FrequencyCoverage(self.trace_set, 2)
        action_pair_function = agilkia.EventPairCoverage(self.trace_set, 2, event_to_str=lambda ev: ev.action)
        objective_functions.extend([frequency_function, action_pair_function])
        ga_optimizer = agilkia.GeneticOptimizer(self.trace_set, objective_functions, num_of_selected_traces=2,
                                                number_of_iterations=50, number_of_chromosomes=50, prob_cross=0.85,
                                                prob_mutate=0.005, elitism=True, elitism_rate=0.2)
        selected_traces, best_objective_value = ga_optimizer.optimize()
        self.assertEqual({self.trace4, self.trace3}, set(selected_traces.traces))
        self.assertEqual(((2 / 3) * 0.5 + (1.5 / 2.6) * 0.5), best_objective_value)

    def test_ga2(self):
        objective_functions = []
        frequency_function = agilkia.FrequencyCoverage(self.trace_set, 1)
        action_status_function = agilkia.EventCoverage(self.trace_set, 1,
                                                       event_to_str=lambda ev: ev.action + "_" + str(ev.status))
        objective_functions.extend([frequency_function, action_status_function])
        ga_optimizer = agilkia.GeneticOptimizer(self.trace_set, objective_functions, num_of_selected_traces=2,
                                                number_of_iterations=50, number_of_chromosomes=50, prob_cross=0.85,
                                                prob_mutate=0.005, elitism=True, elitism_rate=0.2)
        selected_traces, best_objective_value = ga_optimizer.optimize()
        self.assertEqual([self.trace4], selected_traces.traces)
        self.assertEqual(((2 / 3) * 0.5 + (0.8 / 2.6) * 0.5), best_objective_value)

    def test_ga3(self):
        frequency_function = agilkia.FrequencyCoverage(self.trace_set, 1)
        status_function = agilkia.EventCoverage(self.trace_set, 1, event_to_str=lambda ev: str(ev.status))
        objective_functions = [frequency_function, status_function]
        ga_optimizer = agilkia.GeneticOptimizer(self.trace_set, objective_functions, num_of_selected_traces=2,
                                                number_of_iterations=50, number_of_chromosomes=50, prob_cross=0.85,
                                                prob_mutate=0.005, elitism=True, elitism_rate=0.2)
        selected_traces, best_objective_value = ga_optimizer.optimize()
        self.assertEqual([self.trace4], selected_traces.traces)
        self.assertEqual(((2 / 3) * 0.5 + (0.8 / 2.6) * 0.5), best_objective_value)

    def test_ga4(self):
        objective_functions = []
        frequency_function = agilkia.FrequencyCoverage(self.trace_set, 2)
        action_pair_function = agilkia.EventPairCoverage(self.trace_set, 2, event_to_str=lambda ev: ev.action)
        objective_functions.extend([frequency_function, action_pair_function])
        ga_optimizer = agilkia.GeneticOptimizer(self.trace_set, objective_functions, num_of_selected_traces=2,
                                                number_of_iterations=50, number_of_chromosomes=50, prob_cross=0.85,
                                                prob_mutate=0.005, elitism=True, elitism_rate=0.2)
        selected_traces, best_objective_value = ga_optimizer.optimize()
        self.assertEqual({self.trace4, self.trace3}, set(selected_traces.traces))
        self.assertEqual(((2 / 3) * 0.5 + (1.5 / 2.6) * 0.5), best_objective_value)

    def test_ga5(self):
        objective_functions = []
        frequency_function = agilkia.FrequencyCoverage(self.trace_set, 2)
        action_pair_function = agilkia.EventPairCoverage(self.trace_set, 2,
                                                         event_to_str=lambda ev: ev.action + "_" + str(ev.status))
        objective_functions.extend([frequency_function, action_pair_function])
        ga_optimizer = agilkia.GeneticOptimizer(self.trace_set, objective_functions, num_of_selected_traces=2,
                                                number_of_iterations=50, number_of_chromosomes=50, prob_cross=0.85,
                                                prob_mutate=0.005, elitism=True, elitism_rate=0.2)
        selected_traces, best_objective_value = ga_optimizer.optimize()
        self.assertEqual({self.trace4, self.trace3}, set(selected_traces.traces))
        self.assertEqual(((2 / 4) * 0.5 + (1.5 / 2.6) * 0.5), best_objective_value)

    def test_ga6(self):
        objective_functions = []
        frequency_function = agilkia.FrequencyCoverage(self.trace_set, 2)
        action_pair_function = agilkia.EventPairCoverage(self.trace_set, 2, event_to_str=lambda ev: str(ev.status))
        objective_functions.extend([frequency_function, action_pair_function])
        ga_optimizer = agilkia.GeneticOptimizer(self.trace_set, objective_functions, num_of_selected_traces=2,
                                                number_of_iterations=50, number_of_chromosomes=50, prob_cross=0.85,
                                                prob_mutate=0.005, elitism=True, elitism_rate=0.2)
        selected_traces, best_objective_value = ga_optimizer.optimize()
        self.assertEqual({self.trace4, self.trace3}, set(selected_traces.traces))
        self.assertEqual(((2 / 4) * 0.5 + (1.5 / 2.6) * 0.5), best_objective_value)