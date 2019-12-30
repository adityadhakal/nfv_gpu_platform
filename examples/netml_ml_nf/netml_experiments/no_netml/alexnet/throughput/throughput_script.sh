#! /bin/bash
for i in 1 2 4 8 16 32 "ab"
do
    echo $i
    datamash -t, mean 3 pstdev 3 < *_"b${i}_"*.txt
done
