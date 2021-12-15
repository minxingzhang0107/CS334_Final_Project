#!/bin/bash
#SBATCH --job-name=resnet_test
#SBATCH --output=resnet_test.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12

source /local/home/wdai26/anaconda3/bin/activate
cd /local/home/wdai26/Pawpularity/ || exit
conda activate brainnet

python resnet_test.py