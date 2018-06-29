#!/bin/bash
#SBATCH --mail-user=rui.wang.1@vanderbilt.edu
#SBATCH --partition=pascal
#SBATCH --account=biostat_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=2
#SBATCH --ntasks=1
#SBATCH --mem=12G
#SBATCH --time=96:00:00
#SBATCH --output=example2.out

module load Anaconda3
module load CUDA
module load cuDNN
source activate theano_env
python example2.py
