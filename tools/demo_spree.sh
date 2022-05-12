#!/usr/bin/env bash

if [[ $# > 0 ]]
then
  INPUT="$1"
else
  INPUT=spree_logs/log_28_04_2022__14_46_08_edited.json 
fi
OUTPUT="${INPUT%.json}.agilkia.json.gz"

echo "Agilkia reading $INPUT -> $OUTPUT"
set -x 

# cp "$INPUT" .

python csv2agilkia.py --split=session "$INPUT" action=url.uri./-2 status=status_code in.url=url in.method=method in.session=session_id # in.params=params meta.timestamp=timestamp

python view_traces.py --ok 200 $OUTPUT

# TODO when more data
# python choose_regression_tests.py -m action_status,action_pair -o genetic $OUTPUT

# python agilkia2robot.py regression.agilkia.json.gz

