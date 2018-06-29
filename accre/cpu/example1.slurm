#!/bin/bash
#SBATCH --mail-user=rui.wang.1@vanderbilt.edu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=12G
#SBATCH --time=96:00:00
#SBATCH --output=example1.out

module load Anaconda3
source activate theano_env
python example1.py
