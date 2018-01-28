#!/bin/bash
for i in {1..100}
do
echo "Generating QOE test $i"
python main.py >> output.txt
done
