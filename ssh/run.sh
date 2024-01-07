#!/bin/bash
#SBATCH -J test
#SBATCH -p normal
#SBATCH --cpus-per-task=12
#SBATCH --mail-type=all
#SBATCH -N 1
#SBATCH -t 5-00:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=sist_gpu[61]
#SBATCH --output=test.out
#SBATCH --error=test.err

cd /public/home/sunhx/icml_2024/code/deq-master/MDEQ-Vision
python tools/cls_train.py --cfg experiments/cifar/cls_mdeq_TINY.yaml