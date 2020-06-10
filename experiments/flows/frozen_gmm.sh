#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH --gres=gpu:1
#SBATCH --mem=8000  # memory in Mb
#SBATCH --cpus-per-task=8
#SBATCH --time=0-6:00:00
#SBATCH -o output/frozen_gmm_%A.txt  # send stdout to outfile
#SBATCH -e output/frozen_gmm_error_%A.txt  # send stderr to errfile
#SBATCH --partition=apollo

source ~/.bashrc
conda activate deconv

python experiments/flows/mixture_compare.py -g -s -f -k 50 -e 300 -l 0.001 -m 128 results/flows/frozen_gmm_6/frozen_gmm_${SLURM_JOBID}






