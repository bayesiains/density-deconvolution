#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH --gres=gpu:1
#SBATCH --mem=8000  # memory in Mb
#SBATCH --cpus-per-task=8
#SBATCH --time=0-12:00:00
#SBATCH -o output/pretraining_frozen_%A.txt  # send stdout to outfile
#SBATCH -e output/pretraining_frozen_error_%A.txt  # send stderr to errfile
#SBATCH --partition=apollo

source ~/.bashrc
conda activate deconv

python experiments/flows/pretraining.py -f -k 50 -l 0.0001 -e 300 -m 128 results/flows/pretraining_frozen/${SLURM_JOBID}







