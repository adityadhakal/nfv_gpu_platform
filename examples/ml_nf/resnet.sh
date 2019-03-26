#! /bin/bash
instance=$1
./og.sh 10 1 $instance -f /home/adhak001/openNetVM-dev/ml_models/ResNet50_ImageNet_CNTK.model -m 1
