#! /bin/bash

echo "Throughputs"
for i in 10 20 30 40 50 60 70 80 90 100
do
    echo "GPU Percentage: ${i}"
    for j in "${i}p_files"
    do
	#we have list of directories percentage wise
	cd $j
	for k in 1 2 4 8 16 32 
	do
	    #iterate throughp each Batch size
	    echo "Batch Size: ${k}"
	    #echo *"_${k}b.txt"
	    datamash -t, max 3 < *"_${k}b.txt"
	done
	cd ..
    done
done
