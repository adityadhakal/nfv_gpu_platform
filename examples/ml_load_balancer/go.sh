#!/bin/bash

cpu=$1
service=$2
dst=$3
pre_service=$4
print=$5

if [ -z $service ]
then
        echo "$0 [cpu-list] [Service ID] [Destination] [Preffered Service] [PRINT]"
        echo "$0 3 0 --> core 3, Service ID 0"
        echo "$0 3,7,9 1 --> cores 3,7, and 9 with Service ID 1"
        echo "$0 3,7,9 1 1000 --> cores 3,7, and 9 with Service ID 1 and Print Rate of 1000"
        exit 1
fi

if [ -z $dst ]
then
        dst=0
fi
if [ -z $print ]
then
        sudo ./build/basic_nf -l $cpu -n 3 --proc-type=secondary -- -r $service -- -d $dst -l $pre_service
else
        sudo ./build/basic_nf -l $cpu -n 3 --proc-type=secondary -- -r $service -- -d $dst -l $pre_service -p $print
fi
