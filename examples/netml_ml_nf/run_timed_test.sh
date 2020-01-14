#! /bin/bash

#run VGG for a bit
./build/app/bridge -l 14 -n 3 --proc-type=secondary -- -r 2 -n 2 -m 9 -- -a 2 -s 50 > slo_clipper_vs_time_interference/vgg19_time_instance1.txt &
export VGG_PID=$!
echo "Sleeping"
sleep 100
echo "Awake"
#kill $VGG_PID
echo "Killing another NF"
echo $1
kill -9 $1
#sleep 20
#kill -9 $VGG_PID
echo "Finished"
