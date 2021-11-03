import random
import numpy as np
from pathlib import Path
import agilkia
import pandas as pd
import unittest
import pytest


def set_seed():
    random.seed(3)
    np.random.seed(3)


class TestDataGenerator(unittest.TestCase):
    TRAINING = Path("fixtures/1026-steps.json")
    TRACE_SET = Path("fixtures/scanner.json")
    training_data = agilkia.TraceSet.load_from_json(TRAINING)
    trace_set = agilkia.TraceSet.load_from_json(TRACE_SET)

    generate_order = {"Action": "categorical", "Status": "numerical",
                      "sessionID": "categorical",
                      "object": "categorical",
                      "param": "categorical"}

    def test_smart_data_generator(self):
        set_seed()
        generated_trace_set = agilkia.TraceSet([])
        generated_trace_set.set_event_chars(self.trace_set.get_event_chars())
        for tr in self.trace_set:
            events = []
            for ev in tr:
                outputs = {"Status": ev.status}
                events.append(agilkia.Event(ev.action, {}, outputs))
            generated_trace_set.append(agilkia.Trace(events))

        input_generators = [agilkia.SessionGenerator(self.generate_order, current_index=2, prefix="client"),
                            agilkia.CategoricalGenerator(self.generate_order, current_index=3),
                            agilkia.CategoricalGenerator(self.generate_order, current_index=4)]

        for i, generator in enumerate(input_generators):
            generator.fit(self.training_data)

        for i, generator in enumerate(input_generators):
            generator.transform(generated_trace_set)

        archived = agilkia.TraceSet.load_from_json(Path("fixtures/smart_scanner.json"))
        self.assertEqual(generated_trace_set, archived)

    def test_not_smart_data_generator(self):
        set_seed()
        generated_trace_set = agilkia.TraceSet([])
        generated_trace_set.set_event_chars(self.trace_set.get_event_chars())
        for tr in self.trace_set:
            events = []
            for ev in tr:
                outputs = {"Status": ev.status}
                events.append(agilkia.Event(ev.action, {}, outputs))
            generated_trace_set.append(agilkia.Trace(events))

        input_generators = [agilkia.SessionGenerator(self.generate_order, current_index=2, prefix="client"),
                            agilkia.RandomCategoryGenerator(self.generate_order, current_index=3),
                            agilkia.RandomCategoryGenerator(self.generate_order, current_index=4)]

        for i, generator in enumerate(input_generators):
            generator.fit(self.training_data)

        for i, generator in enumerate(input_generators):
            generator.transform(generated_trace_set)

        archived = agilkia.TraceSet.load_from_json(Path("fixtures/not_smart_scanner.json"))
        self.assertEqual(generated_trace_set, archived)

    def test_random_number_generator(self):
        set_seed()
        index = 1
        random_number_generator = agilkia.RandomNumberGenerator(self.generate_order, current_index=index)
        random_number_generator.fit(self.training_data)
        random_number_generator.transform(self.trace_set)
        training_column_data = self.training_data.to_pandas()[list(self.generate_order.keys())[index]]
        generated_column_data = self.trace_set.to_pandas()[list(self.generate_order.keys())[index]]
        self.assertTrue(generated_column_data.min() >= training_column_data.min())
        self.assertTrue(generated_column_data.max() <= training_column_data.max())

    def test_random_category_generator(self):
        set_seed()
        index = 0
        random_category_generator = agilkia.RandomCategoryGenerator(self.generate_order, current_index=index)
        random_category_generator.fit(self.training_data)
        random_category_generator.transform(self.trace_set)
        training_column_data = set(self.training_data.to_pandas()[list(self.generate_order.keys())[index]])
        generated_column_data = self.trace_set.to_pandas()[list(self.generate_order.keys())[index]]
        for action in generated_column_data:
            self.assertTrue(action in training_column_data)

