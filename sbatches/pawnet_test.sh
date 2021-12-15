#!/bin/bash
#SBATCH --job-name=pawnet_test
#SBATCH --output=pawnet_test.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12

source /local/home/wdai26/anaconda3/bin/activate
cd /local/home/wdai26/Pawpularity/ || exit
conda activate brainnet

python main_test.py