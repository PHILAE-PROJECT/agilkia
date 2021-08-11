"""
Optimizer to choose a subset of a set of traces, based on some built-in metrics (objective functions) or custom metrics
(objective functions) provided.

Author: Shane Feng, 2021

"""

from typing import List, Union, Callable
from json_traces import Trace, TraceSet
import numpy as np


class OptimizeTraceSet:
    """
    An abstract class for different optimizers with different optimization algorithms for a given trace set.

    Given a set(or list) of traces, an (or a list of) objective function(s), and the number of traces to be extracted,
    the optimizer runs the optimization algorithms and tries to extract a list of traces that maximize the objective
    values returned by the objective functions, scaled by the number of objective functions (equal weights for each).
    """

    def __init__(self, trace_set: Union[List[Trace], TraceSet],
                 objective_functions: Union[str, List[str], Callable, List[Callable]],
                 number_of_traces: int):
        """ Abstract constructor for a traces optimizer.

        Args:
            trace_set (List[Trace] or TraceSet): A list of traces chosen or output by some kind of model. Alternatively,
                a TraceSet instance can be passed in.
            objective_functions (str or List[str] or Callable or List[Callable]): The objective functions to be used to
                evaluate the chosen subset of traces. If a string or a list of strings are passed in, the optimizer
                uses the built in objective functions if exist. A custom or a list of custom objective functions can be
                passed in and used as well.
                An objective function would take in three arguments, which are the trace set, a binary vector to
                represent the solution (on every position, 1 means selected), and the number of traces wanted to be
                selected. Any external variables needed for the objective function should be global variables that can
                be accessed within the function. This is for the convenience to compute different objective values with
                different objective functions and combine them.
            number_of_traces (int): The number of traces wanted to be selected from the trace set.
                For example, if 10 out of 20 traces in the trace set are wanted to be returned as a test suite, the
                optimizer runs the algorithms and tries to choose 10 traces from the trace set that maximize the
                objective values returned by the objective functions.
        """
        self.trace_set = trace_set.traces if isinstance(trace_set, TraceSet) else trace_set
        self.objective_functions = [objective_functions] if isinstance(objective_functions, str) or isinstance(
            objective_functions, Callable) else objective_functions
        self.number_of_traces = number_of_traces
        objective_functions = []
        # TODO: Add more built in objective functions, consider more situations
        try:
            for objective_function in self.objective_functions:
                if isinstance(objective_function, str) and objective_function == "frequency":
                    objective_functions.append(frequency_coverage_objective_function)
                elif isinstance(objective_function, str):
                    raise ValueError(str, "built in objective function not found")
                elif isinstance(objective_function, Callable):
                    objective_functions.append(objective_function)
                else:
                    raise ValueError("Please provide the name of the built in objective function or your custom "
                                     "objective function implementation")
        except ValueError as error:
            print(error)
        self.objective_functions = objective_functions


def frequency_coverage_objective_function(trace_set: List[Trace], solution: np.ndarray, number_of_traces: int) -> float:
    """ An objective function that calculates the frequency coverage of the traces specified in the solution, out of the
        total trace set frequency.
        Typically, if the traces in the trace set are generated using agilkia.SmartSequenceGenerator, there would be a
        frequency meta data associated with it. The coverage of the frequency of the selected traces, out of the total
        traces in the trace set can be used as a metric to measure the goodness of a solution.
        If the traces are not generated with SmartSequenceGenerator or there is no frequency information, please use
        other objective functions or create your own custom function.

    Args:
        trace_set (List[Trace]): The set of traces, represented in a list
        solution (numpy.ndarray): A binary vector with the same length of the trace set that represent the solution. On
        every position, 1 means the trace at the position is selected, and 0 means not selected.
        number_of_traces (int): The number of traces to be selected as a test suite

    Returns:
        The model coverage percentage of the selected traces on the total frequency of the trace set
    """
    # If the number of selected traces in a solution is more than the number of traces wanted, return negative value
    if np.sum(solution) > number_of_traces:
        return (np.sum(solution) - number_of_traces) * -1
    model_coverage = 0
    total = 0
    for trace in trace_set:
        total += trace.meta_data["freq"]
    for index, value in enumerate(solution.tolist()):
        if value == 1:
            trace = trace_set[index]
            model_coverage += trace.meta_data["freq"]
    return model_coverage / total


def combine_objective_functions(trace_set: List[Trace], solution: np.ndarray, number_of_traces: int,
                                objective_functions: List[Callable]) -> float:
    """ Evaluate the solution with all objective functions passed in, assign an equal weight to them based on the number
        of objective functions passed in and combine them.

    Args:
        trace_set (List[Trace]): The set of traces, represented in a list.
        solution (numpy.ndarray): A binary vector with the same length of the trace set that represent the solution. On
        every position, 1 means the trace at the position is selected, and 0 means not selected.
        number_of_traces (int): The number of traces to be selected as a test suite.
        objective_functions (List[Callable]): A list of objective functions to be used to evaluate the solution. Each of
            them return a objective value between 0 and 1.

    Returns:
        Combined the different objective values with an equal weight based on the number of the objective functions, and
        return it.
    """
    number_of_objective_functions = len(objective_functions)
    total_objective_value = 0
    for objective_function in objective_functions:
        objective_value = objective_function(trace_set, solution, number_of_traces)
        total_objective_value += objective_value * (1 / number_of_objective_functions)
    return total_objective_value


class GreedyOptimizer(OptimizeTraceSet):
    """
    A subclass of OptimizeTraceSet that uses the Greedy Search Algorithm to search for a subset of traces that tries to
    maximize the objective value
    """

    def __init__(self, trace_set: Union[List[Trace], TraceSet],
                 objective_functions: Union[str, List[str], Callable, List[Callable]],
                 number_of_traces: int):
        """Creates an optimizer that uses the Greedy Search Algorithm to search for a subset of traces that tries to
            maximize the objective value

        Args:
            trace_set (List[Trace] or TraceSet): A list of traces chosen or output by some kind of model. Alternatively,
                a TraceSet instance can be passed in.
            objective_functions (str or List[str] or Callable or List[Callable]): The objective functions to be used to
                evaluate the chosen subset of traces. If a string or a list of strings are passed in, the optimizer
                uses the built in objective functions if exist. A custom or a list of custom objective functions can be
                passed in and used as well.
            number_of_traces (int): The number of traces wanted to be selected from the trace set.
                For example, if 10 out of 20 traces in the trace set are wanted to be returned as a test suite, the
                optimizer runs the algorithms and tries to choose 10 traces from the trace set that maximize the
                objective values returned by the objective functions
        """
        super().__init__(trace_set=trace_set, objective_functions=objective_functions,
                         number_of_traces=number_of_traces)

    def greedy_search(self):
        """ Implements the greedy search algorithm and applies it on the trace set passed in.
            It loops through each of the not selected trace so far, adds it to the current solution, evaluates the
            solution and records the best objective value achieved by a solution. After one iteration, it selects the
            trace that results in a solution that achieves the best objective value, and go on to the next iteration.
            The algorithm stops when it reaches the number of traces wanted.

        Returns:
            The algorithm returns the best solution it found and the objective value the solution achieves.
        """
        # TODO: Could the for loops possibly be optimized?
        print("Starting Greedy Search...Selecting", self.number_of_traces, "traces")
        num_of_variables = len(self.trace_set)
        solution = np.zeros(num_of_variables)
        best_objective_value = 0
        best_index = None

        for j in range(self.number_of_traces):
            for i in range(num_of_variables):
                if solution[i] != 1:
                    solution[i] = 1
                    objective_value = combine_objective_functions(self.trace_set, solution, self.number_of_traces,
                                                                  self.objective_functions)
                    if objective_value > best_objective_value:
                        best_objective_value = objective_value
                        best_index = i
                    solution[i] = 0
            solution[best_index] = 1

        selected_traces = []
        for i in range(num_of_variables):
            if solution[i] == 1:
                selected_traces.append(self.trace_set[i])
        return selected_traces, best_objective_value
