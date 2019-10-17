#!/usr/bin/env bash
echo -n "Testing trace_analyser.py..."
OUT=traces1_analysed.txt
python ../trace_analyser.py fixtures/traces1.json >$OUT
diff $OUT expected && echo "  passed" && rm -rf $OUT 

echo -n "Testing trace_generator.py..."
rm -rf out.json
python ../trace_generator.py --seed 1234 -i fixtures/inputs1.csv http://www.soapclient.com/xml/soapresponder.wsdl >soap.log
diff soap.log expected && echo "  passed"
echo "Checking out.json -- just the date should be different:"
diff out.json expected

