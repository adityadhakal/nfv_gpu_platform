#! /bin/bash
for f in *.csv
do
    sed -n -e '/max/!p' $f > clean/"${f}"
done
