#! /bin/bash
for f in *.txt
do
    sed -n -e '/batch_size/p' $f > latency/"${f}.csv"
    sed -n -e '/Measurement_/p' $f > throughput/"${f}.csv"
done
