#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH --gres=gpu:1
#SBATCH --mem=8000  # memory in Mb
#SBATCH --cpus-per-task=8
#SBATCH --time=0-12:00:00
#SBATCH -o output/mixture_compare_%A.txt  # send stdout to outfile
#SBATCH -e output/mixture_compare_error_%A.txt  # send stderr to errfile
#SBATCH --partition=apollo

source ~/.bashrc
conda activate deconv

python experiments/flows/mixture_compare.py -k 1 -e 300 -l 0.0001 -m 128 results/flows/elbo_6/mixture_compare_01_elbo_${SLURM_JOBID}
python experiments/flows/mixture_compare.py -k 10 -e 300 -l 0.0001 -m 128 results/flows/elbo_6/mixture_compare_10_elbo_${SLURM_JOBID}
python experiments/flows/mixture_compare.py -k 25 -e 300 -l 0.0001 -m 128 results/flows/elbo_6/mixture_compare_25_elbo_${SLURM_JOBID}
python experiments/flows/mixture_compare.py -k 50 -e 300 -l 0.0001 -m 128 results/flows/elbo_6/mixture_compare_50_elbo_${SLURM_JOBID}






