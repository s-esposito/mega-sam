#!/bin/bash
# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

DATA_PATH=$1
seq=$2


# Run Raft Optical Flows
# CUDA_VISIBLE_DEVICES=0 python cvd_opt/sea_raft_preprocess_flow.py \
#   --datapath=$DATA_PATH/$seq \
#   --path=cvd_opt/checkpoints/Tartan-C-T-TSKH-spring540x960-M.pth \
#   --cfg=cvd_opt/config/eval/spring-L.json \
#   --scene_name $seq

CUDA_VISIBLE_DEVICES=0 python cvd_opt/raft_preprocess_flow.py \
  --datapath=$DATA_PATH/$seq \
  --model=cvd_opt/checkpoints/raft-things.pth \
  --scene_name $seq \
  --mixed_precision

# Run CVD optmization
CUDA_VISIBLE_DEVICES=0 python cvd_opt/cvd_opt.py \
  --scene_name $seq \
  --w_grad 2.0 --w_normal 5.0

