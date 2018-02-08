#!/bin/bash

#export CUDA_VISIBLE_DEVICES=0

cd classification
./run_final.sh
cd ..

#cd classification3nets
#./run_test.sh
#cd ..

#cd classification5min
#./run_test.sh
#cd ..

cd loadprediction
./run_final.sh
cd ..

cd representation
./run_final.sh
cd ..

cd repair
./run_final.sh
cd ..

#cd repair
#./run_test.sh
#cd ..
