#!/bin/bash
#SBATCH --job-name=exp3
#SBATCH --time=2-23:05:00
#SBATCH --nodes 1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=80GB

export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR


# # e1 random walking
# python e1_random_walking.py --split val --eval 1 --auto_gpu_config 0 -n 1 --num_eval_episodes 2000 --agent obj21 


# e2 random map
# set -x
# srun python e2_random_map.py --split val --eval 1 --auto_gpu_config 0 -n 5 --num_eval_episodes 400 --map_size_cm 2400 --use_gtsem 0

# e3 Frontier based method
set -x
srun python e3_frontier.py --split val --eval 1 --auto_gpu_config 0 -n 5 --num_eval_episodes 400 --map_size_cm 2400 --use_gtsem 1