#! /bin/bash
instance=$1
batch_size=$2
./og.sh 10 1 $instance -f /home/adhak001/openNetVM-dev/ml_models/VGG19_ImageNet_Caffe.model -m 2 -b $batch_size
