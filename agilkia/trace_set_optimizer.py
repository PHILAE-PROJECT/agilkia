"""
Optimizer to choose a subset of a set of traces, based on some built-in metrics (objective functions) or custom metrics
(objective functions) provided.

Author: Shane Feng, 2021

"""
import math
import random
import numpy as np
import numpy
from typing import List, Union, Callable, Tuple, Optional
from agilkia.json_traces import TraceSet


class ObjectiveFunction:
    """
    An abstract class for objective functions that can evaluate a solution based on the chosen metrics.

    If a user aims to implement a custom objective function and use it to evaluate the solution, create a subclass of
    this abstract class. See more instruction in the documentation of the constructor and evaluate method below.
    """

    def __init__(self, weight: float = 1.0):
        """Constructor for an objective function.
        Set an empty trace set and 0 selected traces. Set the weight of this objective function.
        Use the set_data method below to set the trace set and selected traces, with more flexibility.

        Args:
            weight (float): Weight of this objective function.
        """
        if not weight > 0:
            raise ValueError(f"Weight has to be a positive number, not {weight}")
        self.weight = weight
        self.trace_set = None
        self.select = 0

    def set_data(self, trace_set: TraceSet, select: int):
        """Set the trace set and number of selected traces.
        With this method, more flexibility is provided. This objective function can be used on different trace set and
        different number of selected traces.
        For specific objective function, pre-compute all necessary data in this method.

        Args:
            trace_set (TraceSet): A set of traces chosen or output by some kind of model.
            select (int): The number of traces wanted to be selected from the trace set.
                For example, if 10 out of 20 traces in the trace set are wanted to be returned as a test suite, the
                optimizer runs the algorithms and tries to choose 10 traces from the trace set that maximize the
                objective values returned by the objective functions.
        """
        if not type(trace_set) == TraceSet:
            raise ValueError(f"Trace set must be a TraceSet object, not {type(trace_set)}")
        if not select > 0:
            raise ValueError(f"Select must be positive, not {select}")
        if not type(select) == int:
            raise ValueError(f"Select must be an integer, not {type(select)}")
        if not select <= len(trace_set):
            raise ValueError(
                f"Select must be smaller than the number of traces in the trace set, length of trace set "
                f"is {len(trace_set)}, select is {select}")
        self.trace_set = trace_set
        self.select = select

    def evaluate(self, solution: numpy.ndarray) -> float:
        """
        Evaluate the solution and return an objective value. Implement in subclass

        Args:
            solution (numpy.ndarray): A binary vector with the same length of the trace set that represent the solution.
                On every position, 1 means the trace at the position is selected, and 0 means not selected.

        Returns: The objective value of the solution
        """
        return 0


class FrequencyCoverage(ObjectiveFunction):
    """
    An objective function that calculates the frequency coverage of the traces selected in the solution, out of the
    total trace set frequency.

    Typically, if the traces in the trace set are generated using agilkia.SmartSequenceGenerator, there would be a
    frequency meta data associated with it. The coverage of the frequency of the selected traces, out of the total
    traces in the trace set can be used as a metric to measure the goodness of a solution.
    If the traces are not generated with SmartSequenceGenerator or there is no frequency information, please use
    other objective functions or create your own custom function.
    """

    def __init__(self, weight: float = 1.0):
        """Constructor for a frequency coverage objective function.
        Set an empty trace set and 0 selected traces. Set the weight of this objective function.
        Set an empty frequencies and total frequency coverage.
        Use the set_data method below to set the trace set, selected traces, frequencies and total frequency coverage
        with more flexibility.

        Args:
            weight (float): Weight of this objective function.
        """
        super().__init__(weight)
        self.frequencies = None
        self.total_frequency_coverage = None

    def set_data(self, trace_set: TraceSet, select: int):
        """Set the trace set, number of selected traces, frequencies and total frequency coverage.
        With this method, more flexibility is provided. This objective function can be used on different trace set and
        different number of selected traces.
        Pre-calculates the frequency coverage for each trace in the trace set and store it in a list.
        Pre-calculates the total frequency coverage.
        If there is no frequency information, throw an exception.

        Args:
            trace_set (TraceSet): A set of traces chosen or output by some kind of model.
            select (int): The number of traces wanted to be selected from the trace set.
        """
        super().set_data(trace_set, select)
        self.frequencies = np.array([trace.get_meta("freq", 0) for trace in trace_set])
        self.total_frequency_coverage = sum(self.frequencies)
        if not self.total_frequency_coverage != 0:
            raise ValueError("There is no frequency information of the traces")

    def evaluate(self, solution: numpy.ndarray) -> float:
        """
        Evaluate the solution based on the frequency coverage of the traces selected in the solution, out of the
        total trace set frequency.

        Args:
            solution (numpy.ndarray): A binary vector with the same length of the trace set that represent the solution.
                On every position, 1 means the trace at the position is selected, and 0 means not selected.

        Returns:
            The percentage of the frequency coverage of the selected traces out of the total frequency of the trace set
        """
        # If the number of selected traces in a solution is more than the number of traces wanted, return negative value
        if np.sum(solution) > self.select:
            return (np.sum(solution) - self.select) * -1
        solution_frequency_coverage = sum(np.array(self.frequencies) * solution)
        return solution_frequency_coverage / self.total_frequency_coverage


class EventCoverage(ObjectiveFunction):
    """
    An objective function that calculates the coverage of the individual events of the selected traces in the
    solution, out of all the traces in the trace set. Typical usage would be action coverage, status coverage and action
    status coverage. If an Event does not have a status in the output, status would be 0.
    """

    def __init__(self, weight: float = 1.0, event_to_str: Optional[Callable] = None):
        """
        Constructor for the event coverage objective function.
        Set an empty trace set and 0 selected traces. Set the weight of this objective function.
        Set an empty trace coverage and total coverage.
        Use the set_data method below to set the trace set, selected traces, trace coverage and total coverage with more
        flexibility.

        Pass in a lambda function to extract the data on the events in the trace

        To calculate action coverage, pass in event_to_string=lambda ev: ev.action.
        To calculate status coverage, pass in event_to_string=lambda ev: str(ev.status).
        To calculate action_status coverage, pass in event_to_string = lambda ev: ev.action + "_" + str(ev.status).

        Args:
            weight (float): Weight of this objective function.
            event_to_str (Callable): The function used to extract data on the events in the trace.
        """
        super().__init__(weight)
        self.event_to_str = event_to_str
        if self.event_to_str is None:
            self.event_to_str = (lambda ev: ev.action)
        self.trace_coverage = []
        self.total_coverage = set()

    def set_data(self, trace_set: TraceSet, select: int):
        """Set the trace set, number of selected traces and pre-compute the trace coverage and total coverage.
        With this method, more flexibility is provided. This objective function can be used on different trace set and
        different number of selected traces.

        Args:
            trace_set (TraceSet): A set of traces chosen or output by some kind of model.
            select (int): The number of traces wanted to be selected from the trace set.
        """
        super().set_data(trace_set, select)
        self.trace_coverage = []
        self.total_coverage = set()
        for trace in trace_set:
            trace_coverage = set(trace.action_counts(event_to_str=self.event_to_str).keys())
            self.trace_coverage.append(trace_coverage)
            self.total_coverage = self.total_coverage.union(trace_coverage)
        self.trace_coverage = np.array(self.trace_coverage)

    def evaluate(self, solution: numpy.ndarray) -> float:
        """
        Evaluate the solution based on the selected coverage of the selected traces in the solution,
        out of all the traces in the trace set.

        Args:
            solution (numpy.ndarray): A binary vector with the same length of the trace set that represent the solution.
                On every position, 1 means the trace at the position is selected, and 0 means not selected.

        Returns:
            The percentage of the selected coverage of the selected traces out of the total action status
                coverage of the trace set
        """
        if np.sum(solution) > self.select:
            return (np.sum(solution) - self.select) * -1
        solution_action_status_coverage = set()
        solution = np.array(solution, dtype=bool)
        for trace_coverage in self.trace_coverage[solution]:
            solution_action_status_coverage = solution_action_status_coverage.union(trace_coverage)
        return len(solution_action_status_coverage) / len(self.total_coverage)


class EventPairCoverage(ObjectiveFunction):
    """
    An objective function that calculates the coverage of the event pairs of the selected traces in the solution,
    out of all the traces in the trace set. Typical usage would be action pair coverage, status pair coverage and
    action status pair coverage. If an Event does not have a status in the output, status would be 0.
    For example, for a trace, the events in a trace are "unlock, scan, scan, checkout". The action pairs would be
    "unlock_scan, scan_scan, scan_checkout".
    """

    def __init__(self, weight: float = 1.0, event_to_str: Optional[Callable] = None):
        """
        Constructor for the event pair coverage objective function.
        Set an empty trace set and 0 selected traces. Set the weight of this objective function.
        Set an empty trace coverage and total coverage.
        Use the set_data method below to set the trace set, selected traces, trace coverage and total coverage with more
        flexibility.

        Pass in a lambda function to extract the data on the events in the trace

        To calculate action pair coverage, pass in event_to_string=lambda ev: ev.action.
        To calculate status pair coverage, pass in event_to_string=lambda ev: str(ev.status).
        To calculate action_status pair coverage, pass in event_to_string = lambda ev: ev.action + "_" + str(ev.status).

        Args:
            weight (float): Weight of this objective function.
            event_to_str (Callable): The function used to extract data on the events in the trace.
        """
        super().__init__(weight)
        self.event_to_str = event_to_str
        if self.event_to_str is None:
            self.event_to_str = (lambda ev: ev.action)
        self.trace_coverage = []
        self.total_coverage = set()

    def set_data(self, trace_set: TraceSet, select: int):
        """Set the trace set, number of selected traces and pre-compute the trace coverage and total coverage.
        With this method, more flexibility is provided. This objective function can be used on different trace set and
        different number of selected traces.

        Args:
            trace_set (TraceSet): The set of traces.
            select (int): The number of traces to be selected as a test suite.
        """
        super().set_data(trace_set, select)
        self.trace_coverage = []
        self.total_coverage = set()
        for trace in trace_set:
            trace_coverage = set()
            for i in range(len(trace) - 1):
                trace_coverage.add(self.event_to_str(trace[i]) + "_" + self.event_to_str(trace[i + 1]))
            self.trace_coverage.append(trace_coverage)
            self.total_coverage = self.total_coverage.union(trace_coverage)
        self.trace_coverage = np.array(self.trace_coverage)

    def evaluate(self, solution: numpy.ndarray) -> float:
        """
        Evaluate the solution based on the selected coverage of the selected traces in the solution,
        out of all the traces in the trace set.

        Args:
            solution (numpy.ndarray): A binary vector with the same length of the trace set that represent the solution.
                On every position, 1 means the trace at the position is selected, and 0 means not selected.

        Returns:
            The percentage of the selected coverage of the selected traces out of the total action status
                coverage of the trace set
        """
        if np.sum(solution) > self.select:
            return (np.sum(solution) - self.select) * -1
        solution_action_pair_coverage = set()
        solution = np.array(solution, dtype=bool)
        for trace_coverage in self.trace_coverage[solution]:
            solution_action_pair_coverage = solution_action_pair_coverage.union(trace_coverage)
        return len(solution_action_pair_coverage) / len(self.total_coverage)



class ClusterCoverage(ObjectiveFunction):
    """
    An objective function that calculates how many clusters are covered by the selected traces.
    """

    def __init__(self, weight: float = 1.0):
        """
        Constructor for the cluster coverage objective function.

        Args:
            weight (float): Weight of this objective function.
        """
        super().__init__(weight)
        self.max_coverage = 1

    def set_data(self, trace_set: TraceSet, select: int):
        """Set the trace set and the number of desired traces.

        This throws an exception if the given ``trace_set`` is not clustered.

        Args:
            trace_set (TraceSet): A set of traces chosen or output by some kind of model.
            select (int): The number of traces wanted to be selected from the trace set.
        """
        super().set_data(trace_set, select)
        if not trace_set.is_clustered():
            raise ValueError("cluster coverage requires a trace set that is clustered")
        self.max_coverage = trace_set.get_num_clusters()

    def evaluate(self, solution: numpy.ndarray) -> float:
        """
        Evaluate the cluster coverage of the selected traces.

        Args:
            solution (numpy.ndarray): A binary vector with the same length of the trace set that represent the solution.
                On every position, 1 means the trace at the position is selected, and 0 means not selected.

        Returns:
            The fraction of clusters covered by the selected traces.
        """
        if np.sum(solution) > self.select:
            return (np.sum(solution) - self.select) * -1
        cluster_num = self.trace_set.get_clusters()
        coverage = set()
        for i,s in enumerate(solution):
            if s:
                coverage.add(cluster_num[i])
        return len(coverage) / self.max_coverage



class TraceSetOptimizer:
    """
    An abstract class for different optimizers with different optimization algorithms for a given trace set.

    Given a trace set, an (or a list of) objective function(s), and the number of traces to be extracted,
    the optimizer runs the optimization algorithms and tries to extract a set of traces that maximize the objective
    values returned by the objective functions, scaled by the number of objective functions (equal weights for each).
    """

    def __init__(self, objective_functions: Union[ObjectiveFunction, List[ObjectiveFunction]]):
        """
        Constructor for a trace set optimizer.
        Set an empty trace set and 0 selected traces. Use set_data method below to set the trace set and the number of
        selected traces.

        Args:
            objective_functions (ObjectiveFunction or List[ObjectiveFunction]): The objective functions to be used to
                evaluate the chosen subset of traces. Built in functions or custom defined functions are acceptable.
                Use ObjectiveFunction class to create your custom objective function. See more in the documentation of
                that class.
        """
        self.verbose = True   # TODO: let the caller turn this on/off
        self.trace_set = None
        self.objective_functions = [objective_functions] if not isinstance(objective_functions,
                                                                           list) else objective_functions
        self.select = 0
        for objective_function in self.objective_functions:
            if not isinstance(objective_function, ObjectiveFunction):
                raise ValueError(
                    "Please provide valid built in objective functions or your custom objective functions "
                    "created using the ObjectiveFunction class")

    def set_data(self, trace_set: TraceSet, select: int):
        """Set the trace set and the number of selected traces for the optimizer and the objective functions passed in

        Args:
            trace_set (TraceSet): A set of traces chosen or output by some kind of model.
            select (int): The number of traces wanted to be selected from the trace set.
                For example, if 10 out of 20 traces in the trace set are wanted to be returned as a test suite, the
                optimizer runs the algorithms and tries to choose 10 traces from the trace set that maximize the
                objective values returned by the objective functions.
        """
        if not type(trace_set) == TraceSet:
            raise ValueError(f"Trace set must be a TraceSet object, not {type(trace_set)}")
        if not select > 0:
            raise ValueError(f"Select must be positive, not {select}")
        if not type(select) == int:
            raise ValueError(f"Select must be an integer, not {type(select)}")
        if not select <= len(trace_set):
            raise ValueError(
                f"Select must be smaller than the number of traces in the trace set, length of trace set "
                f"is {len(trace_set)}, select is {select}")
        for obj_func in self.objective_functions:
            obj_func.set_data(trace_set, select)
        self.trace_set = trace_set
        self.select = select

    def objective(self, solution: np.ndarray) -> float:
        """
        Evaluate the solution with all objective functions passed in and use the associated weights combine them.

        Args:
            solution (numpy.ndarray): A binary vector with the same length of the trace set that represent the solution.
                On every position, 1 means the trace at the position is selected, and 0 means not selected.

        Returns:
            Combined the different objective values with an equal weight based on the number of the objective
                functions, and return it. Result would be between 0 and 1
        """
        total_objective_value = 0
        total_weight = 0
        for obj_func in self.objective_functions:
            total_weight += obj_func.weight
            total_objective_value += obj_func.evaluate(solution) * obj_func.weight
        return total_objective_value / total_weight


class GreedyOptimizer(TraceSetOptimizer):
    """
    A subclass of TraceSetOptimizer that uses the Greedy Search Algorithm to search for a subset of traces that tries to
    maximize the objective value
    """

    def __init__(self, objective_functions: Union[ObjectiveFunction, List[ObjectiveFunction]]):
        """Creates an optimizer that uses the Greedy Search Algorithm to search for a subset of traces that tries to
        maximize the objective value.
        Set an empty trace set and 0 selected traces. Use set_data method to set the trace set and the number of
        selected traces.

        Args:
            objective_functions (ObjectiveFunction or List[ObjectiveFunction]): The objective functions to be used to
                evaluate the chosen subset of traces. Built in functions or custom defined functions are acceptable.
                Use ObjectiveFunction class to create your custom objective function. See more in the documentation of
                that class.
        """
        super().__init__(objective_functions=objective_functions)

    def optimize(self) -> Tuple[TraceSet, float]:
        """ Implements the greedy search algorithm and applies it on the trace set passed in.
        It loops through each of the not selected trace so far, adds it to the current solution, evaluates the
        solution and records the best objective value achieved by a solution. After one iteration, it selects the
        trace that results in a solution that achieves the best objective value, and go on to the next iteration.
        The algorithm stops when it reaches the number of traces wanted.

        Returns:
            The algorithm returns the best trace set it found and the objective value the solution achieves.
        """
        if self.verbose:
            print(f"Starting Greedy Search with max traces={self.select}")
        num_of_variables = len(self.trace_set)
        solution = np.zeros(num_of_variables)
        best_objective_value = 0
        best_index = None

        for j in range(self.select):
            if self.verbose:
                print(f"  iter={j} best={best_objective_value}")
            for i in range(num_of_variables):
                if solution[i] != 1:
                    solution[i] = 1
                    objective_value = self.objective(solution)
                    if objective_value > best_objective_value:
                        best_objective_value = objective_value
                        best_index = i
                    solution[i] = 0
            solution[best_index] = 1
        selected_traces = [self.trace_set[i] for i in range(num_of_variables) if solution[i]]
        selected_traces = TraceSet(selected_traces)
        return selected_traces, best_objective_value


class ParticleSwarmOptimizer(TraceSetOptimizer):
    """
    A subclass of TraceSetOptimizer that uses the Particle Swarm Optimization Algorithm to search for a subset of
    traces that tries to maximize the objective value
    """

    def __init__(self, objective_functions: Union[ObjectiveFunction, List[ObjectiveFunction]],
                 num_of_particles: int = 400, num_of_iterations: int = 500, c1: float = 2.0, c2: float = 2.0):
        """Creates an optimizer that uses the Particle Swarm Optimization Algorithm to search for a subset of traces
        that tries to maximize the objective value. This is using the Binary version of PSO.
        Set an empty trace set and 0 selected traces. Use set_data method to set the trace set, number of
        selected traces and the hyper parameters.

        Args:
            objective_functions (Callable or List[Callable]): The objective functions to be used to
                evaluate the chosen subset of traces. Built in functions or custom defined functions are acceptable.
                An objective function would take in three arguments, which are the trace set, a binary vector to
                represent the solution (on every position, 1 means selected), and the number of traces wanted to be
                selected. Any external variables needed for the objective function should be global variables that can
                be accessed within the function. This is for the convenience to compute different objective values with
                different objective functions and combine them.
            num_of_particles (int): Number of particles used in the algorithm
            num_of_iterations (int): Number of iterations of the algorithm
            c1 (float): Controlling parameter of the influence of the particle's personal best position on the
                particle's velocity during update
            c2 (float): Controlling parameter of the influence of the population's global best position on the
                particle's velocity during update
        """
        super().__init__(objective_functions=objective_functions)
        if not num_of_particles > 0:
            raise ValueError(f"The number of particles should be positive, not {num_of_particles}")
        if not type(num_of_particles) == int:
            raise ValueError(f"The number of particles should be integer, not {type(num_of_particles)}")
        if not num_of_iterations > 0:
            raise ValueError(f"The number of iterations should be positive, not {num_of_iterations}")
        if not type(num_of_iterations) == int:
            raise ValueError(f"The number of iterations should be integer, not {type(num_of_particles)}")
        if not c1 > 0:
            raise ValueError(f"c1 should be positive, not {c1}")
        if not c2 > 0:
            raise ValueError(f"c2 should be positive, not {c2}")
        self.num_of_iterations = num_of_iterations
        self.num_of_particles = num_of_particles
        self.c1 = c1
        self.c2 = c2
        self.num_of_variables = 0
        self.upper_bound = None
        self.lower_bound = None

    def set_data(self, trace_set: TraceSet, select: int):
        """Set the trace set and the number of selected traces of the optimizer and the objective functions passed in
        Set the num of particles, num of iterations, c1 and c2

        Args:
            trace_set (TraceSet): A set of traces chosen or output by some kind of model.
            select (int): The number of traces wanted to be selected from the trace set.
                For example, if 10 out of 20 traces in the trace set are wanted to be returned as a test suite, the
                optimizer runs the algorithms and tries to choose 10 traces from the trace set that maximize the
                objective values returned by the objective functions.
        """
        super().set_data(trace_set, select)
        self.num_of_variables = len(trace_set)
        # Every bit in the solution can only be either 0 or 1
        self.upper_bound = np.ones(self.num_of_variables)
        self.lower_bound = np.zeros(self.num_of_variables)

    def optimize(self) -> Tuple[TraceSet, float]:
        """ Implements the Particle Swarm Optimization Algorithm and applies it on the trace set passed in.
        A particle's position is a solution that the particle holds. The algorithm initialise the positions to be a
        binary vector that holds 0 and 1 randomly. The velocity if used to update the position of a particle. A particle
        also holds a personal best position and a personal best value that the particle has been to and achieved. The
        population holds a global best position and a global best value.
        During iterations, the particle compares it's current objective value with the personal best value, and
        substitute the global best position with the current position if it's current value is higher. It would also do
        the same to the global best position and global best value.
        During velocity update, w, c1, c2 are controlling parameters to control the influence of current velocity,
        personal best and global best on the update. After update, if the velocity exceeds to maximum or minimum,
        set it to maximum mor minimum velocity.
        After updating velocity, transform the velocity vector into sigmoid values. Generate a random vector between 0
        and 1. If the sigmoid values are larger than the generated values at an index position, change it to 1. This
        final vector will be the particle's new position.

        Returns:
            The algorithm returns the best trace set it found and the objective value the solution achieves.
        """
        if self.verbose:
            print(f"Starting Particle Swarm Optimizer with particles={self.num_of_particles} iterations={self.num_of_iterations}" +
                f" c1={self.c1}, c2={self.c2}, max traces={self.select}")

        # Define the upper bound and lower bound of the controlling parameter of the influence for the previous
        # velocity on the particle's velocity during update
        w_max = 0.9
        w_min = 0.1

        # Define maximum and minimum velocity to avoid exceeding this limit. The value 6 is suggested by the author of
        # the algorithm
        v_max = np.ones(self.num_of_variables) * 6
        v_min = -v_max

        # Initialise the particle container, global best position, and global best objective value
        particles = []
        gbest_x = np.zeros(self.num_of_variables)
        gbest_val = -math.inf

        # Initialise the particle population
        for i in range(self.num_of_particles):
            particle = {}
            # Round positions to nearest integers
            particle['X'] = np.rint(
                np.add(np.subtract(self.upper_bound, self.lower_bound) * np.random.rand(self.num_of_variables),
                       self.lower_bound))
            # Velocity
            particle['V'] = np.zeros(self.num_of_variables)
            # Position of a particle, this is the solution the particle holds
            particle['PBESTX'] = np.zeros(self.num_of_variables)
            # Objective value of the solution
            particle['PBESTO'] = -math.inf
            particles.append(particle)

        # Start iteration
        for t in range(self.num_of_iterations):
            if self.verbose and t % 10 == 0:
                print(f"  iter={t} best={gbest_val:.4f}")
            # Update personal best and global best
            for index, particle in enumerate(particles):
                current_x = particle['X']
                particle["O"] = self.objective(current_x)

                if particle['O'] > particle['PBESTO']:
                    particle['PBESTX'] = current_x
                    particle['PBESTO'] = particle["O"]

                if particle['O'] > gbest_val:
                    gbest_x = current_x
                    gbest_val = particle['O']

            # Update particle's position and velocity
            # w would change adaptively according the current iteration
            w = w_max - t * ((w_max - w_min) / self.num_of_iterations)

            for index, particle in enumerate(particles):
                particle['V'] = w * particle['V'] + \
                                self.c1 * np.random.rand(self.num_of_variables) * np.subtract(particle['PBESTX'],
                                                                                              particle['X']) + \
                                self.c2 * np.random.rand(self.num_of_variables) * np.subtract(gbest_x, particle['X'])

                # Check min max velocity
                index_greater = np.where(particle['V'] > v_max)[0].tolist()
                index_smaller = np.where(particle['V'] < v_min)[0].tolist()
                if index_greater:
                    for idx in index_greater:
                        particle['V'][idx] = v_max[idx]
                if index_smaller:
                    for idx in index_smaller:
                        particle['V'][idx] = v_min[idx]

                # Sigmoid transfer
                sigmoid = 1 / (1 + np.exp(-particle['V']))
                temp = np.random.rand(self.num_of_variables) < sigmoid
                temp = temp * 1
                particle['X'] = temp
        selected_traces = [self.trace_set[i] for i in range(self.num_of_variables) if gbest_x[i]]
        selected_traces = TraceSet(selected_traces)
        return selected_traces, gbest_val


class GeneticOptimizer(TraceSetOptimizer):
    """
    A subclass of TraceSetOptimizer that uses the Genetic Algorithm to search for a subset of traces that tries to
    maximize the objective value
    """

    def __init__(self, objective_functions: Union[ObjectiveFunction, List[ObjectiveFunction]],
                 num_of_iterations: int = 500, num_of_chromosomes: int = 400, prob_cross: float = 0.85,
                 prob_mutate: float = 0.005, elitism_rate: float = 0.2, crossover: str = "double"):
        """Creates an optimizer that uses the Genetic Algorithm to search for a subset of traces that tries to maximize
        the objective value.
        
        Set an empty trace set and 0 selected traces. Use set_data method to set the trace set, number of
        selected traces and hyper parameters.

        Args:

            objective_functions (Callable or List[Callable]): The objective functions to be used to
                evaluate the chosen subset of traces. Built in functions or custom defined functions are acceptable.
                An objective function would take in three arguments, which are the trace set, a binary vector to
                represent the solution (on every position, 1 means selected), and the number of traces wanted to be
                selected. Any external variables needed for the objective function should be global variables that can
                be accessed within the function. This is for the convenience to compute different objective values with
                different objective functions and combine them.
            num_of_iterations (int): Number of iterations of the algorithm.
            num_of_chromosomes (int): Number of solutions in the population.
            prob_cross (float): probability of crossover.
            prob_mutate (float): probability of mutate.
            elitism_rate (float): Rate of elitism.
            crossover (str): The method used to crossover. Choose between double and single.
        """
        super().__init__(objective_functions)
        if not (crossover == "single" or crossover == "double"):
            raise ValueError(f"Crossover method should only be single or double, not {crossover}")
        if not (0 < prob_mutate < 1):
            raise ValueError(f"Probability of mutate must be between 0 < ... < 1, not {prob_mutate}")
        if not (0 < prob_cross < 1):
            raise ValueError(f"Probability of crossover must be between 0 < ... < 1, not {prob_cross}")
        if not num_of_iterations > 0:
            raise ValueError(f"The number of iterations must be positive, not {num_of_iterations}")
        if not type(num_of_iterations) == int:
            raise ValueError(f"The number of iterations should be an integer, not {type(num_of_iterations)}")
        if not num_of_chromosomes > 0:
            raise ValueError(f"The number of chromosomes must be positive, not {num_of_chromosomes}")
        if not type(num_of_chromosomes) == int:
            raise ValueError(f"The number of chromosomes should be an integer, not {type(num_of_chromosomes)}")
        if not (0 <= elitism_rate < 1):
            raise ValueError(f"The elitism rate must be between 0 <= ... < 1, not {elitism_rate}")
        self.num_of_iterations = num_of_iterations
        self.num_of_chromosomes = num_of_chromosomes + (num_of_chromosomes % 2)  # make it even!
        self.prob_mutate = prob_mutate
        self.prob_cross = prob_cross
        self.elitism_rate = elitism_rate
        self.crossover_method = crossover
        self.population = None
        self.new_population = None
        self.num_of_genes = 0

    def set_data(self, trace_set: TraceSet, select: int):
        """Set the trace set and the number of selected traces of the optimizer and the objective functions passed in
        Set the num of chromosomes, num of iterations, probability of crossover, probability of mutate and elitism rate.

        Args:
            trace_set (TraceSet): A set of traces chosen or output by some kind of model.
            select (int): The number of traces wanted to be selected from the trace set.
                For example, if 10 out of 20 traces in the trace set are wanted to be returned as a test suite, the
                optimizer runs the algorithms and tries to choose 10 traces from the trace set that maximize the
                objective values returned by the objective functions.
        """
        super().set_data(trace_set, select)
        self.num_of_genes = len(trace_set)
        # Initialise population
        self.population = np.rint(np.random.rand(self.num_of_chromosomes, self.num_of_genes))

    def _normalise_objective_values(self) -> numpy.ndarray:
        """After initialising the population of solutions, there might be some solutions that have a negative objective
        value because of the random initialisation. To be able to select parents using the roulette_wheel method below,
        we need to have all objective values to be not negative. We normalise the objective values so that they are all
        between 0 and 1.

        This reads self.objective_values and returns a normalized version of those probabilities, which
        sums to 1.0.  It does not write to any self fields.
 
        Returns:
            The normalised objective values.
        """
        min_objective_value = min(self.objective_values)
        if min_objective_value < 0:
            temp = self.objective_values - min_objective_value + 1
        else:
            temp = self.objective_values
        normalised_objective_values = temp / np.sum(temp)
        return normalised_objective_values

    def _roulette_wheel(self, normalised_objective_values: numpy.ndarray) -> int:
        """Calculate the cumulative sum of the objective values and output the selected index based on probability

        Args:
            normalised_objective_values (numpy.ndarray): Normalised objective values of the population, which are all
                between 0 and 1.

        Returns:
            Selected index.
        """
        cum_sum = np.cumsum(normalised_objective_values)
        r = random.random()
        for index, condition in enumerate(r <= cum_sum):
            if condition:
                return index

    def _select_parents(self, population: numpy.ndarray, normalised_objective_values: numpy.ndarray) -> numpy.ndarray:
        """Use the roulette_wheel method to select 2 solutions from the population as parents. The selected parents are
        passed in to crossover method.

        This is a pure function of its inputs - it does not read or update any self fields.

        Args:
            population (numpy.ndarray): The population of solutions
            normalised_objective_values (numpy.ndarray): The normalised objective values of the solutions.

        Returns:
            The selected solutions as parents.
        """
        # TODO: improve this to use np.default_rng Generator
        # return random.choices(population, normalised_objective_values, k=2)
        selected_parents_indexes = []
        selected_parents = []
        for i in range(2):
            index = self._roulette_wheel(normalised_objective_values)
            while index in selected_parents_indexes:
                index = self._roulette_wheel(normalised_objective_values)
            selected_parents_indexes.append(index)
            selected_parents.append(population[index])
        selected_parents = np.array(selected_parents)
        return selected_parents

    def _crossover(self, parent1: numpy.ndarray, parent2: numpy.ndarray) -> \
            Tuple[numpy.ndarray, numpy.ndarray]:
        """Exchange a subset of the two parents and produce new solutions. With single point crossover, randomly pick a
        point in the solution and exchange the other part. With double point crossover, randomly pick two points in the
        solution and exchange two different parts. After crossover, use probability to decide whether to keep the
        changed solution or not.

        This never mutates the input parent1 or parent2.  It returns fresh child arrays, not aliased
        with the parent arrays.

        Args:
            parent1 (numpy.ndarray): A solution used to crossover.
            parent2 (numpy.ndarray): A solution used to crossover.

        Returns:
            Either the changed solutions or a copy of the original solutions based on probability.
        """
        child1 = None
        child2 = None
        if self.crossover_method == "single":
            crossover_point = random.randint(1, self.num_of_genes - 2)
            child1 = np.concatenate([parent1[0:crossover_point], parent2[crossover_point: self.num_of_genes]])
            child2 = np.concatenate([parent2[0:crossover_point], parent1[crossover_point: self.num_of_genes]])

        elif self.crossover_method == "double":
            crossover_point1 = random.randint(1, self.num_of_genes - 2)
            crossover_point2 = random.randint(1, self.num_of_genes - 2)
            while crossover_point1 == crossover_point2:
                crossover_point2 = random.randint(1, self.num_of_genes - 2)
            if crossover_point1 > crossover_point2:
                temp = crossover_point1
                crossover_point1 = crossover_point2
                crossover_point2 = temp
            child1 = np.concatenate([parent1[0:crossover_point1], parent2[crossover_point1: crossover_point2],
                                     parent1[crossover_point2:self.num_of_genes]])
            child2 = np.concatenate([parent2[0:crossover_point1], parent1[crossover_point1: crossover_point2],
                                     parent2[crossover_point2:self.num_of_genes]])

        r1 = random.random()
        child1 = child1 if r1 <= self.prob_cross else parent1.copy()  # copy, because we may mutate child
        r2 = random.random()
        child2 = child2 if r2 <= self.prob_cross else parent2.copy()
        return child1, child2

    def _mutate(self, child: numpy.ndarray):
        """After crossover, for every bit in the solution, flip the bit (0 to 1 or 1 to 0) based on probability.

        This destructively mutates the input child array.

        Args:
            child (numpy.ndarray): Solution returned by crossover method.

        Returns:
            Solution with bits flipped based on probability.
        """
        for i in range(self.num_of_genes):
            r = random.random()
            if r <= self.prob_mutate:
                child[i] = not child[i]

    def _add_elites(self, new_pop: np.ndarray, new_pop_objective_values: np.ndarray) -> np.ndarray:
        """ After crossover and mutate, before entering the next generation, keep the best solutions from last
        generation which have the highest objective values. The number of elites are controlled by the elitism rate.

        This reads ``self.population`` but does not update any self fields.

        Returns:
            The next generation of population.
        """
        number_of_elites = int(self.num_of_chromosomes * self.elitism_rate)
        # we make a temporary copy of the objective values, so we can mutate it.
        old_objective_values = self.objective_values.copy()

        for i in range(number_of_elites):
            max_index = np.argmax(old_objective_values)
            elite = self.population[max_index]
            max_value = old_objective_values[max_index]
            assert max_value == self.objective(elite)
            min_index = np.argmin(new_pop_objective_values)
            min_value = new_pop_objective_values[min_index]
            if min_value < max_value:
                # assert new_pop_objective_values[min_index] == self.objective(new_pop[min_index])
                new_pop[min_index, :] = elite
                new_pop_objective_values[min_index] = max_value
                # to 'delete' (hide) the elite entry and ensure it will not be picked again,
                # we just set its objective value to -inf so that it will never be max. 
                old_objective_values[max_index] = - np.inf
            else:
                break
        return new_pop, new_pop_objective_values

    def optimize(self):
        """Implements the Genetic Algorithm and applies it on the trace set passed in.
        The algorithms initialise the population. Each chromosome is a solution with the length of the trace set. Each
        bit in the chromosome indicates the trace in the trace set is selected or not.
        In each iteration, two parents are selected from the population based on probability. Two children are produced
        by applying crossover on the parents, controlled by probability of crossover. The children mutate based on
        probability and controlled by probability of mutate. The mutated children are added to the new population, which
        are also randomly initialised. If elitism is enabled, apply elitism to keep the best solutions from last
        generation, controlled by the elitism rate. Keep track of the best solution found so far and enter the next
        iteration (generation).

        Returns:
            The algorithm returns the best trace set it found and the objective value the solution achieves.
        """
        if self.verbose:
            print(f"Starting Genetic Algorithm with chromosomes={self.num_of_chromosomes} iterations={self.num_of_iterations}," +
                f" mutate={self.prob_mutate}, crossover={self.prob_cross},{self.crossover_method}, elitism={self.elitism_rate}," +
                f" max traces={self.select}")
        self.objective_values = np.apply_along_axis(self.objective, 1, self.population)
        best = np.max(self.objective_values)
        for i in range(self.num_of_iterations):
            new_best = np.max(self.objective_values)
            if self.elitism_rate > 0:
                assert new_best >= best  # we should always get better (at least no worse)
            best = new_best
            # progress message
            if self.verbose and i % 10 == 0:
                print(f"  iter={i} best={best:.4f}")

            new_population = np.zeros((self.num_of_chromosomes, self.num_of_genes))
            normalised_objective_values = self._normalise_objective_values()
            for j in range(0, self.num_of_chromosomes, 2):
                # Selection
                [parent1, parent2] = self._select_parents(self.population, normalised_objective_values)
                # Crossover
                child1, child2 = self._crossover(parent1, parent2)
                # Mutation
                self._mutate(child1)
                self._mutate(child2)
                new_population[j, :] = child1
                new_population[j + 1, :] = child2
            assert new_population.shape == self.population.shape
            # objective function is expensive, so we try to calculate it only once per iteration!
            new_objective_values = np.apply_along_axis(self.objective, 1, new_population)
            if self.elitism_rate > 0:
                new_population, new_objective_values = self._add_elites(new_population, new_objective_values)
            self.population = new_population
            self.objective_values = new_objective_values
        best_index = np.argmax(self.objective_values)
        best_objective_value = self.objective_values[best_index]
        solution = self.population[best_index]
        selected_traces = [self.trace_set[i] for i in range(self.num_of_genes) if solution[i]]
        selected_traces = TraceSet(selected_traces)
        return selected_traces, best_objective_value
