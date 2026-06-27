#!/bin/bash
#SBATCH -J pi05_min
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/logs/pi05-min-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/logs/pi05-min-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 0-00:10:00

echo "hello"
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
