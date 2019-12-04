#! /bin/bash
echo " Mean Batch Size, Median Latency (micro-sec)"
for f in *.txt
do
    echo $f
    datamash -t, --header-in mean 2 median 6 <$f
done

    
