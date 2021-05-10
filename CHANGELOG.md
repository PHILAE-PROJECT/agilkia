# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.0] - 2021-05-10
### Added
- PrefixTraceExtractor supports more flexible ways of extracting ML features from traces.
- Added eShop example: trace file and some scripts 
- Updated dependency versions to better support Python 3.8 and higher.
- Added developers_agilkia.yml so it is easy to create an Agilkia development environment.
- Added CHANGELOG.md

### Changed
- Restructured the main docs page so it is more useful on ReadTheDocs.io.
- Improved the docs of TracePrefixExtractor to show new features.
- Added --cluster option to read_scanette_csv.py script, which does bag-of-words MeanShift clustering.

### Removed
- Nothing.


## [0.6.0] - 2020-03-30
### Added
- Support for hierarchical clustering in output JSON files.  See TraceSet.set_clusters etc.
- Added example of hierarchical clustering in examples/scanner/cluster_hierarchical.py.
- Add examples/scanner helper scripts for converting to/from Scanette *.csv files.
- Add examples/scanner scripts to illustrate clustering and choosing a given number of tests.
- Added Event.status_float for non-integer status values, with NaN for missing/strange values.

### Changed
- Improved the tests for reading various older versions of JSON files.
- Improve API of TraceSet.with_traces_grouped_by to take a lambda function for the key, 
  and to throw exception on missing keys by default.
- Cluster labels are now stored in the JSON file, with an optional scipy linkage array for hierarchical clusters.


## [0.5.1] - 2019-12-11
### Changed
- Bug fix release to correct dependencies.  Removed fastcluster package as it clashed with some numpy versions.


## [0.5.0] - 2019-12-11
### Added
- Added examples/scanner for the SuperMarket scanner example (scanette in French).
- Nicer printing of Event objects.
- Added TracePrefixExtractor (an sklearn feature extractor for trace prefixes) to support machine learning.
- Added SmartSequenceGenerator that uses an ML model to generate sequences of actions.
- Added SmartSequenceGenerator.generate_all_traces algorithm, to generate all high-probability traces.

### Changed
- Allow SmartSequenceGenerator to take URLs, and add a way of executing generated tests.
- Fixed errors in decode_outputs, to handle zeep objects properly.
- Change TraceSet.clusters to be private (._clusters) so it is not saved.  
  This is for consistency, since we cannot easily save _cluster_data and both are needed for visualize_clusters.
  The idea is that clustering is transient, so needs to be redone after save/load.
- Improved the default copying of meta_data when a TraceSet is created with traces all from one parent.


## [0.4.2] - 2019-10-28
### Added
- Added support for readthedocs so that documentation is built automatically for each version.

### Changed
- Added consistent meta_data field to all three levels: Event, Trace, TraceSet. 
- Made clustering and visualisation a bit more flexible.


## [0.3.1] - 2019-10-20
### Added
- Add more meta-data to the flit pyproject.toml file.

### Changed
- Delay import of arff until user calls to_arff, because liac-arff is pip-only and conda-build does not seem to handle that.
- Documentation in Sphinx format, to go onto readthedocs.org.


## [0.3.0] - 2019-10-19
### Added
- Added clustering and visualization of clusters using TSNE.
- Added TraceSet.with_traces_split and with_traces_grouped_by methods to support some common cases.

### Changed
- Major refactoring to use class Event, instead of just dict. 
- Also started using mypy type checker and removed most type errors.


## [0.2.2] - 2019-10-16
### Added
- Added initial support for smart test generation, using ML action prediction model.
- Add username + password support when interacting with a web service.
- Added save_to_arff method to support the StackedTrees tool from Gilles Bisson.
- Added traces_to_pandas to convert to a Pandas DataFrame.  Includes action, inputs, and output status and errors.

### Changed
- Switched to MIT license and added more docs.
- Moved action_chars into meta-data.
- Implemented some JSON-trace-version upgrade code, so older data files are read correctly.
- Major refactoring of lists-of-list-of-events into TraceSet and Trace objects, with GDF-like meta-data.
- Color non-zero status results red in trace_to_string to make them stand out.
