#! /bin/bash
echo " Median Throughput (ips), Median Latency (micro-sec)"
for f in *.txt
do
    echo $f
    datamash -t, --header-in median 1 median 3 <$f
done

    
