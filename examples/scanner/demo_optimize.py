from agilkia import TraceSet, EventCoverage, EventPairCoverage, GreedyOptimizer, ParticleSwarmOptimizer, GeneticOptimizer
from pathlib import Path

traces = TraceSet.load_from_json(Path("1026-steps.split.json"))
metrics = [EventCoverage(event_to_str=lambda ev: f"{ev.action}_{ev.status}"), 
           EventPairCoverage()]

optimizer = GreedyOptimizer(metrics)  # 9 traces, 100.0% coverage
#optimizer = ParticleSwarmOptimizer(metrics)  # 10 traces, 97.5% coverage
#optimizer = GeneticOptimizer(metrics)  # 10 traces, 100.0% coverage
optimizer.set_data(traces, 10)
tests,coverage = optimizer.optimize()
print(f"optimized to {tests} tests, {coverage*100.0:.1f}% coverage")
tests.save_to_json(Path("1026-steps-optimized.agilkia.json"))

