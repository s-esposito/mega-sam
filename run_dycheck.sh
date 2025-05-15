#!/bin/bash

DATA_PATH=datasets/dycheck

evalset=(
    # apple
    # backpack
    # block
    # creeper
    # handwavy
    # haru-sit
    # mochi-high-five
    # pillow
    # spin
    # sriracha-tree
    teddy
    # paper-windmill
)


echo "Dataset path: $DATA_PATH"

for seq in ${evalset[@]}; do
    echo "Sequence: $seq"
    bash ./mono_depth_scripts/run_mono-depth_dycheck.sh $DATA_PATH $seq
    bash ./tools/evaluate_dycheck.sh $DATA_PATH $seq
    bash ./cvd_opt/cvd_opt_dycheck.sh $DATA_PATH $seq
    # bash ./tools/visualize_demo.sh $DATA_PATH $seq
done
