#!/bin/bash
#SBATCH --job-name=llm_hm
#SBATCH --time=2-23:05:00
#SBATCH --nodes 1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=20GB

export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

# hm3d
set -x
srun python main_llm.py --split val --eval 1 --auto_gpu_config 0 \
-n 1 --num_eval_episodes 2000 --num_processes_on_first_gpu 1 \
--load pretrained_models/llm_model.pt  --use_gtsem 0 \
--num_local_steps 10 --exp_name exp_llm


# Gibson
# set -x
# srun python main_llm.py \
# --split val \
# --eval 1 \
# --auto_gpu_config 0 \
# -n 1 \
# --num_eval_episodes 1000 \
# --num_processes_on_first_gpu 1 \
# --load pretrained_models/llm_model.pt  \
# --use_gtsem 0 \
# --num_local_steps 10 \
# --exp_name exp_llm_gibson \
# --task_config tasks/objectnav_gibson.yaml \
# --map_size_cm 2400
