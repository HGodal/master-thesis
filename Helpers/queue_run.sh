#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100 
#SBATCH --time=24:00:00
#SBATCH --job-name=GPU_rum
#SBATCH --output=out.out
 
# Activate environment
uenv miniconda3-py39
conda activate gan

# Run the Python script that uses the GPU
python -u python_file.py