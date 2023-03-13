#!/bin/bash
#SBATCH --job-name=llm_dg
#SBATCH --time=2-12:05:00
#SBATCH --nodes 1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB

python data_generator.py