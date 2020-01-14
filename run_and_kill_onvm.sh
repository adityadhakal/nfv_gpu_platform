#! /bin/bash

for gpu_percent in 2 4 6 10 20 30 40 50 60 70 80 90 100
do
    echo "Starting percent "$gpu_percent
    for batch in 1 2 4 8 16 32
    do
	echo "Starting batch: "$batch
	#first enter the Onvm Directory
	cd onvm
	#now run openNetVM with right arguments
	./onvm_mgr/x86_64-native-linuxapp-gcc/onvm_mgr -l 0,2,4,6,8 -n 4 --proc-type=primary -- -p 0xf -s stdout &
	ONVM_PID=$!

	#now sleep for 30 seconds because you want the manager to load
	sleep 8

	#now turn on the NF
	cd /home/adhak001/dev/openNetVM_sameer/examples/netml_ml_nf
	./build/app/bridge -l 10 -n 3 --proc-type=secondary -- -r 1 -n 1 -m 8 -- -g $gpu_percent -b $batch > mobilenet_profile/"batch_${batch}_percent_${gpu_percent}.txt" &
	NF_PID=$!
	sleep 60

	kill -9 $NF_PID
	kill -9 $ONVM_PID
	pkill bridge
	cd /home/adhak001/dev/openNetVM_sameer
    done
    echo $gpu_percent" Complete"
done
