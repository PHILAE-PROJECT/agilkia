Example Analysis of Supermarket Scanner Tests (Scanette)
========================================================

Scanette is a simulated supermarket scanner application, where customers enter
a supermarket, choose products, and can purchase them.

The Scanette source code is publically available from: ???

This folder contains several useful scripts:

 * read_scanette_csv.py               = convert Scanette *.csv files into Agilkia *.json files.
 * choose_regression_tests.py         = cluster tests and choose N regression tests / cluster.
 * write_scanette_csv.py              = convertAgilkia *.json files back into Scanette *.csv.
 * cluster_hierarchical.py            = cluster tests hierarchically and show dendrogram.
 * generate_missing_tests_scanette.py = generate tests for some missing clusters.
 * analyse_scanette2.py               = cluster traces and compare against Scanette system tests.
 * test_scanner.py     = pyunit tests to check that the scripts are working.


Example run::

  python read_scanette_csv.py --split 1026-steps.csv
  python choose_regression_tests.py 1026-steps.split.json
  python write_scanette_csv.py regression_tests.json


Example of running the chosen regression tests on the Scanette implementation
(change ';' to ':' for Linux/Mac)::

  SCANETTE=~/git/scanette   # path to your copy of the Scanette repository.
  cp regression_tests.csv "$SCANETTE/replay"
  pushd "$SCANETTE/replay"
  CP='ScanetteTestReplay.jar;json-simple.jar;junit-4.12.jar'
  java -cp "$CP;scanette.jar" fr.philae.ScanetteTraceExecutor regression_tests.csv

  # to run against all the Scanette mutants (using bash)
  for M in scanette-mu*.jar
  do
      echo "Testing mutant $M"
      java -cp "$CP;$M" fr.philae.ScanetteTraceExecutor regression_tests.csv
  done
  popd
