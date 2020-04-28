#! /bin/bash

for f in *.txt
do
    echo $f
    cat $f | sed -n '/Measurement_/p' | tail -n 20 | datamash -t , mean 4 pstdev 4 mean 6 pstdev 6
done
