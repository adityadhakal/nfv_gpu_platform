#! /bin/bash
for batch_size in 1 2 4 8 16 32
do
    echo "======================= Batch Size "$batch_size" ======================================="
    for percent in 2 4 6 10 20 30 40 50 60 70 80 90 100
    do
	FILE_NAME="batch_${batch_size}_percent_${percent}.txt.csv"
	echo "Batch Size: "$batch_size" Percent "$percent
	datamash -t, median 2 median 6 median 12 < $FILE_NAME
    done
done


    
