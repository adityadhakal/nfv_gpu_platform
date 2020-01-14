#! /bin/bash

for i in clipper time
do
    for f in $i*
    do
	echo "#"$f
	datamash -t, median 2 pstdev 2 median 12 pstdev 12 < $f
    done
done
