#! /bin/bash

echo $RTE_TARGET
for gpu_percent in 2 4 6 10 20 30 40 50 60 70 80 90 100
do
    pkill bridge
    pkill onvm_mgr
    echo "Starting percent "$gpu_percent
    for batch in 1 2 4 8 16 32
    do
	echo "Starting batch: "$batch
	#first enter the Onvm Directory
	#cd onvm
	#now run openNetVM with right arguments
	./go.sh 0,2,4,6,8 0xf -s stdout
	ONVM_PID=$!

	#now sleep for 30 seconds because you want the manager to load
	#sleep 8

	#now turn on the NF
	
	/home/adhak001/dev/openNetVM_sameer/examples/netml_ml_nf/build/app/bridge -l 10 -n 3 --proc-type=secondary -- -r 1 -n 1 -m 8 -- -g $gpu_percent -b $batch >/home/adhak001/dev/openNetVM_sameer/examples/netml_ml_nf/mobilenet_profile/"batch_${batch}_percent_${gpu_percent}.txt" &
	NF_PID=$!
	echo "NF PID "$NF_PID
	sleep 60

	kill -9 $NF_PID
	kill -9 $ONVM_PID
	pkill bridge
#	cd /home/adhak001/dev/openNetVM_sameer
    done
    echo $gpu_percent" Complete"
done
