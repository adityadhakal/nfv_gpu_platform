#! /bin/bash
for i in 1 2 4 8 16 32 "ab"
do
    echo $i
    datamash -t, mean 6 pstdev 6 mean 10 pstdev 10 < *_"b${i}_"*.txt
done
