#!/bin/bash
#SBATCH --job-name=expv1%j        # name of job
#SBATCH --output=exp_outputs/exp_retina_TravailGPU%j.out
#SBATCH --error=exp_outputs/exp_retina_TravailGPU%j.err
#SBATCH --time=30:00:00       # Maximum execution time (HH:MM:SS)
#SBATCH --nodes=1             # Number of nodes
#SBATCH --gpus=1              # Number of GPUs
#SBATCH --cpus-per-task=8     # Number of CPU cores per task
#SBATCH --partition=V100-32GB       # GPU partition (you might need to adjust this)

set -x

# Load the required modules (you may need to adjust this based on your environment)
# module load cuda
# module load cudnn
module load python
# Activate your Conda environment
conda activate bifdet

# Navigate to the directory containing your Python script
cd /home/ids/akeshavarzi/projects/bifdet2024/retinanet_pipeline/
file_paths_args = "--exp_path / \
                   --model_weights_path / \
                   --annot_fname bboxes_16_cases.json"
training_args="--patch_size 256\
               --val_patch_size 256\
               --patience 60 \
               --optimizer SGD \
               --training \
               --max_epochs 500 \
               --batch_size 1 \
               --val_interval 1 \
               --amp \
               --cache_ds \
               --detection_per_img 200 \
               --nms_thresh 0.22 \
               --score_thresh_glb 0.1 \
               --detector_lr 1e-3 \
               --w_cls 0.5 \
               --topk_candidates_per_level 1000 \
               --sw_inferer_overlap 0.25 \
               --sw_batch_size \
               --sw_mode constant \
               --num_cands 4 \
               --center_in_gt \
               --batch_size_per_image 64 \
               --positive_fraction 0.5 \
               --pool_size 20 \
               --min_neg 16 \
               --momentum 0.9 \
               --weight_decay 3e-5 \
               --nesterov \
               --a_scheduler_step_size 150 \
               --a_scheduler_gamma 0.1 \
               --wu_scheduler_multiplier 1 \
               --wu_scheduler_total_epoch 10"
python -u training.py ${training_args} ${file_paths_args}
