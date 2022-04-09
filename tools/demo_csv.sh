#!/usr/bin/env bash

INPUT=../examples/scanner/1026-steps.csv 

echo "Agilkia tools demo on a copy of $INPUT"
set -x 

cp "$INPUT" .

python csv2agilkia.py --split=sessionID --cluster 1026-steps.csv  action=4  in.sessionID=2  in.object=3  in.param=5.rmbrackets.nonempty  meta.sequence=0.int   meta.timestamp=1.msec2iso   status=6.nonquestion.float

python view_traces.py 1026-steps.agilkia.json.gz

python choose_regression_tests.py -m action_status,action_pair -o genetic 1026-steps.agilkia.json.gz

python agilkia2robot.py regression.agilkia.json.gz

