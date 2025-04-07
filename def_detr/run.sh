#!/bin/bash

file_paths_args="--output_dir / \
                 --annot_fname BifDet_lbl0_min_10.json"

training_args="--patch_size 256 \
               --epochs 200 \
               --batch_size 1 \
               --num_queries 300 \
               --pos_weight 10 \
               --cache_ds \
               --weight_decay 0.0001 \
               --a_scheduler_step_size 80 \
               --a_scheduler_gamma 0.1 \
               --wu_scheduler_multiplier 1 \
               --wu_scheduler_total_epoch 10"

python -u training.py ${training_args} ${file_paths_args}