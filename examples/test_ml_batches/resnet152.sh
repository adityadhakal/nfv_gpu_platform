#! /bin/bash
instance=$1
batch_size=$2
./og.sh 10 1 $instance -f /home/adhak001/openNetVM-dev/ml_models/ResNet152_ImageNet_CNTK.model -m 3 -b $batch_size
