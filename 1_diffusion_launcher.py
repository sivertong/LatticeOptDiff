import torch as th
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

os.environ['TOPODIFF_LOGDIR'] = './checkpoints/diff_logdir'

TRAIN_FLAGS="--batch_size 32 --save_interval 20000 --use_fp16 True"
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"

DATA_FLAGS="--data_dir ./data/dataset_1_diff/training_data"

%run scripts/image_train.py $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $DATA_FLAGS