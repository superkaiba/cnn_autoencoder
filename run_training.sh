#!/bin/sh
#SBATCH --partition=main,lab-bengioy
#SBATCH --gres=gpu:32gb:1                                    # a100l for 80GB, a100 for 40GB
#SBATCH --mem=8G                                       # Ask for 10 GB of RAM
#SBATCH --time=6:00:00                                 # The job will run for 3 hours
#SBATCH -o /home/mila/t/thomas.jiralerspong/scratch/cnn_autoencoder/slurm-%j.out  # Write the log on scratch

module load anaconda/3
conda init
conda activate cnn_autoencoder
wandb enabled
export LOGDIR=/home/mila/t/thomas.jiralerspong/scratch/cnn_autoencoder

python /home/mila/t/thomas.jiralerspong/delta_ai/CNN_autoencoder/train.py \
     --architecture "original_wider_less_channels" \
     --log_dir $LOGDIR 