#! /bin/bash
instance=$1
batch_size=$2
./og.sh 10 1 $instance -f /home/adhak001/openNetVM-dev/ml_models/ResNet50_ImageNet_CNTK.model -m 1 -b $batch_size &
#sleep 5
#nvidia-smi --query-gpu=timestamp,utilization.gpu --format=csv -lms 200 > "resnet_gpu_utilization_batch_${2}.dat"
