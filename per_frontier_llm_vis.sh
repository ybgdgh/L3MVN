#!/bin/bash
#SBATCH --job-name=llm_hm_without_low_score
#SBATCH --time=2-23:05:00
#SBATCH --nodes 1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=v100:1
#SBATCH --mem=50GB
#SBATCH –-output=objectnav

export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR


mkdir $TMPDIR/myArchive
tar xvzf /scratch/%USER/Objectnav/hm3d.tar.gz $TMPDIR/myArchive
# cd $TMPDIR

module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
source $HOME/.envs/habitat21/bin/activate

# hm3d
set -x
srun python main_llm_vis.py --split val --eval 1 --auto_gpu_config 0 \
-n 8 --num_eval_episodes 250 --num_processes_on_first_gpu 8 \
--load pretrained_models/llm_model.pt  --use_gtsem 0 \
--num_local_steps 10 --exp_name exp_llm_without_low_score
# --global_downscaling 4

cp -r outputdata /scratch/$USER/mydataset/



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
