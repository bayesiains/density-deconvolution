#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH --gres=gpu:1
#SBATCH --mem=8000  # memory in Mb
#SBATCH --cpus-per-task=8
#SBATCH --time=0-12:00:00
#SBATCH -o output/pretraining_%A.txt  # send stdout to outfile
#SBATCH -e output/pretraining_error_%A.txt  # send stderr to errfile
#SBATCH --partition=apollo

source ~/.bashrc
conda activate deconv

python experiments/flows/pretraining.py -k 50 -l 0.0001 -e 300 results/flows/pretraining_2/${SLURM_JOBID}







