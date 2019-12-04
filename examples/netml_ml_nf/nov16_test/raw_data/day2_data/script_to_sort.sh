#! /bin/bash

#read all the file names
for f in *.txt
do
    echo $f
    sed -n -e "/batch_size/!p" $f>throughput_values/$f
    sed -n -e "/batch_size/p" $f>latency_and_batch_size/$f
done
