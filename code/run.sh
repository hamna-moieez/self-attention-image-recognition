#!/bin/bash

#SBATCH -J selfattention
#SBATCH -o out.txt
#SBATCH -p gpu-all
#SBATCH --gres gpu:1
#SBATCH -c 4
#SBATCH --mem 100000MB
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=hamna21797@gmail.com

module load slurm cuda10.1/toolkit

python -u trainer.py

