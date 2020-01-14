#! /bin/bash

for f in *.csv
do
    echo $f
    datamash -t, median 2 pstdev 2 median 12 pstdev 12 <$f
done
