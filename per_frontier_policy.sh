#!/bin/bash
#SBATCH --job-name=hm3dnopool
#SBATCH --time=2-23:05:00
#SBATCH --nodes 1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=80GB

export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

set -x
srun python main_frontier_policy.py --total_num_scenes 6 --auto_gpu_config 0 \
--split train --eval 0 --exp_name exp_f_policy_hm3d_nopool --num_local_steps 15


# set -x
# srun python main_frontier_policy.py --split val --eval 1 --auto_gpu_config 0 \
# -n 5 --num_eval_episodes 200 --num_processes_on_first_gpu 5 \
# --load pretrained_models/model_f_policy.pth --exp_name exp_f_policy_val --use_gtsem 0
# --print_images 1 -d /data/p305574/data/results/ 