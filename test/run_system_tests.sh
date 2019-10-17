#!/usr/bin/env bash
echo -n "Testing trace_analyser.py..."
OUT=traces1_analysed.txt
python ../trace_analyser.py fixtures/traces1.json >$OUT
diff $OUT expected && echo "  passed" && rm -r $OUT 

echo -n "Testing trace_generator.py..."
rm -r out.json 2>/dev/null
LOG=soap.log
python ../trace_generator.py --seed 1234 -i fixtures/inputs1.csv http://www.soapclient.com/xml/soapresponder.wsdl >$LOG
diff $LOG  expected && echo "  passed" && rm -f $LOG
echo "Checking out.json -- just the date should be different:"
diff out.json expected
# comment this next line out if you want to preserve out.json:
rm -f out.json

