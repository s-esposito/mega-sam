#!/bin/bash

DATA_PATH=datasets/davis/JPEGImages/Full-Resolution

evalset=(
    car-turn
    hike
)


echo "Dataset path: $DATA_PATH"

for seq in ${evalset[@]}; do
    echo "Sequence: $seq"
    bash ./mono_depth_scripts/run_mono-depth_demo.sh $DATA_PATH $seq
    bash ./tools/evaluate_demo.sh $DATA_PATH $seq
    bash ./cvd_opt/cvd_opt_demo.sh $DATA_PATH $seq
    bash ./tools/visualize_demo.sh $DATA_PATH $seq
done
