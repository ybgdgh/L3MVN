#!/bin/bash
#SBATCH --job-name=llm_hm_without_low_score
#SBATCH --time=2-23:05:00
#SBATCH --nodes 1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=50GB

export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

# hm3d
set -x
srun python main_llm_vis.py --split val --eval 1 --auto_gpu_config 0 \
-n 8 --num_eval_episodes 250 --num_processes_on_first_gpu 8 \
--load pretrained_models/llm_model.pt  --use_gtsem 0 \
--num_local_steps 10 --exp_name exp_llm_without_low_score
# --global_downscaling 4


# Gibson
# set -x
# srun python main_llm_vis.py \
# --split val \
# --eval 1 \
# --auto_gpu_config 0 \
# -n 5 \
# --num_eval_episodes 200 \
# --num_processes_on_first_gpu 5 \
# --load pretrained_models/llm_model.pt  \
# --use_gtsem 0 \
# --num_local_steps 10 \
# --exp_name exp_llm_gibson \
# --task_config tasks/objectnav_gibson.yaml \
# --map_size_cm 3600
