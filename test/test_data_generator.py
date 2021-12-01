import random
import numpy as np
from pathlib import Path
import agilkia
import pandas as pd
import unittest
import pytest


THIS_DIR = Path(__file__).parent


def set_seed():
    random.seed(3)
    np.random.seed(3)



class TestDataGenerator(unittest.TestCase):
    TRAINING = Path(THIS_DIR / "fixtures" / "1026-steps.json")
    TRACE_SET = Path(THIS_DIR / "fixtures" / "scanner.json")
    training_data = agilkia.TraceSet.load_from_json(TRAINING)
    trace_set = agilkia.TraceSet.load_from_json(TRACE_SET)

    generate_order = {"Action": "categorical", "Status": "numerical",
                      "sessionID": "categorical",
                      "object": "categorical",
                      "param": "categorical"}


    def similar_freqs(self, traces1, traces2, column, delta=0.025):
        """Checks if two trace sets have a similar frequency distribution for a given input/output column."""
        freq1 = traces1.to_pandas()[column].value_counts(normalize=True)
        freq2 = traces2.to_pandas()[column].value_counts(normalize=True)
        # print(f"similar freq on {set(freq1.index)} & {set(freq2.index)}")
        for val in set(freq1.index) & set(freq2.index):   # intersection should include all common values
            # print(f"  {val:16s} {freq1[val]:.6f} {freq2[val]:.6f}  ratio={freq1[val] / freq2[val]:.6f}  diff={freq1[val] - freq2[val]:.6f}")
            assert freq1[val] == pytest.approx(freq2[val], abs=delta), f"column={column} value={val}"

    def test_smart_data_generator(self):
        set_seed()
        generated_trace_set = agilkia.TraceSet([])
        generated_trace_set.set_event_chars(self.trace_set.get_event_chars())
        for tr in self.trace_set:
            events = [agilkia.Event(ev.action, {}, {"Status": ev.status}) for ev in tr]
            generated_trace_set.append(agilkia.Trace(events))
        input_generators = [agilkia.SessionGenerator(self.generate_order, current_index=2, prefix="client"),
                            agilkia.CategoricalGenerator(self.generate_order, current_index=3),
                            agilkia.CategoricalGenerator(self.generate_order, current_index=4)]

        for i, generator in enumerate(input_generators):
            generator.fit(self.training_data)

        for i, generator in enumerate(input_generators):
            generator.transform(generated_trace_set)

        # save the new generated trace set, in case we want to manually compare it?
        generated_trace_set.save_to_json(Path(THIS_DIR / "fixtures" / "smart_scanner_new.json"))
        archived = agilkia.TraceSet.load_from_json(Path(THIS_DIR / "fixtures" / "smart_scanner.json"))
        self.similar_freqs(archived, generated_trace_set, "object")
        self.similar_freqs(archived, generated_trace_set, "param")

        # This checks if the traces are all identical, but that is too strong.
        # same = generated_trace_set.equal_traces(archived)
        # if not same:
        #     print(f"  trace lengths: {len(generated_trace_set.traces)} =?= {len(archived.traces)}")
        #     for (a,b) in zip(generated_trace_set.traces, archived.traces):
        #         print(f"  trace: {a} =?= {b}  {a.equal_events(b)}")
        # self.assertTrue(same)

    def test_not_smart_data_generator(self):
        set_seed()
        generated_trace_set = agilkia.TraceSet([])
        generated_trace_set.set_event_chars(self.trace_set.get_event_chars())
        for tr in self.trace_set:
            events = [agilkia.Event(ev.action, {}, {"Status": ev.status}) for ev in tr]
            generated_trace_set.append(agilkia.Trace(events))
        input_generators = [agilkia.SessionGenerator(self.generate_order, current_index=2, prefix="client"),
                            agilkia.RandomCategoryGenerator(self.generate_order, current_index=3),
                            agilkia.RandomCategoryGenerator(self.generate_order, current_index=4)]

        for i, generator in enumerate(input_generators):
            generator.fit(self.training_data)

        for i, generator in enumerate(input_generators):
            generator.transform(generated_trace_set)

        # save the new generated trace set, in case we want to manually compare it?
        generated_trace_set.save_to_json(Path(THIS_DIR / "fixtures" / "not_smart_scanner_new.json"))
        archived = agilkia.TraceSet.load_from_json(Path(THIS_DIR / "fixtures" / "not_smart_scanner.json"))
        self.similar_freqs(archived, generated_trace_set, "object")
        self.similar_freqs(archived, generated_trace_set, "param")

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

