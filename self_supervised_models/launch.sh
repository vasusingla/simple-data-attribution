#!/bin/bash
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=4
#SBATCH --mem=16G

python train_ssl.py