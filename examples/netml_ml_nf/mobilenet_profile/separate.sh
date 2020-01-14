#! /bin/bash
for file in *.txt
do
    sed -n -e '/batch_size/p' $file > latency/$file.csv
done
