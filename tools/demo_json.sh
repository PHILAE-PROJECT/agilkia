#!/usr/bin/env bash

INPUT=spree.json

echo "Agilkia tools demo on $INPUT"
set -x 

# cp "$INPUT" .

python csv2agilkia.py --split=session spree.json action=action status=status in.path=path in.session=session_id in.params=params meta.timestamp=timestamp

python view_traces.py spree.agilkia.json.gz

# TODO when more data
# python choose_regression_tests.py -m action_status,action_pair -o genetic spree.agilkia.json.gz

# python agilkia2robot.py regression.agilkia.json.gz

