#!/bin/bash

file_paths_args="--exp_path / \
                   --model_weights_path / \
                   --annot_fname BifDet_lbl0_min_10.json"

training_args="--patch_size 256\
               --val_patch_size 256\
               --patience 60 \
               --optimizer SGD \
               --training \
               --max_epochs 200 \
               --batch_size 1 \
               --val_interval 1 \
               --amp \
               --cache_ds \
               --detection_per_img 300 \
               --nms_thresh 0.22 \
               --score_thresh_glb 0.1 \
               --detector_lr 1e-2 \
               --w_cls 0.5 \
               --topk_candidates_per_level 1000 \
               --sw_inferer_overlap 0.25 \
               --sw_batch_size \
               --sw_mode constant \
               --num_cands 8 \
               --center_in_gt \
               --batch_size_per_image 64 \
               --positive_fraction 0.5 \
               --pool_size 20 \
               --min_neg 16 \
               --momentum 0.9 \
               --weight_decay 3e-5 \
               --nesterov \
               --a_scheduler_step_size 50 \
               --a_scheduler_gamma 0.1 \
               --wu_scheduler_multiplier 1 \
               --wu_scheduler_total_epoch 10"
python -u training.py ${training_args} ${file_paths_args}
