#!/bin/bash
#
#SBATCH --job-name=motionmillion-download
#SBATCH --output=/projects/illinois/eng/cs/shenlong/personals/haoyu/imu-humans/.log/motionmillion-download.txt
#
#SBATCH --mail-user=haoyuyh3@illinois.edu
#SBATCH --mail-type=ALL
#
#SBATCH --partition=shenlong2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2-0:00
#
#SBATCH --chdir=/u/haoyuyh3

source ~/.bashrc
conda activate imu-humans

cd /projects/illinois/eng/cs/shenlong/personals/haoyu/imu-humans

# git clone https://huggingface.co/datasets/InternRobotics/MotionMillion
git clone https://haoyuhsu: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@huggingface.co/datasets/InternRobotics/MotionMillion
cd MotionMillion
git lfs pull