#!/bin/bash
#
#SBATCH --job-name=texts
#SBATCH --output=/projects/illinois/eng/cs/shenlong/personals/haoyu/imu-humans/.log/texts.txt
#
#SBATCH --mail-user=haoyuyh3@illinois.edu
#SBATCH --mail-type=ALL
#
#SBATCH --partition=shenlong2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2-0:00
#
#SBATCH --chdir=/u/haoyuyh3

source ~/.bashrc
conda activate imu-humans

cd /projects/illinois/eng/cs/shenlong/personals/haoyu/imu-humans/motionmillion_scripts
python get_texts.py