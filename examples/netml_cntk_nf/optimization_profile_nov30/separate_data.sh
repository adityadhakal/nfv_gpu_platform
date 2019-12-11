#! /bin/bash

#read all the files
for files in *.txt
do
    echo $files
    sed -n -e "/batch_size/p" $files > latency_values/$files
    sed -n -e "/Measurement_interval/p" $files >throughput_values/$files
done