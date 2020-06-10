#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH --gres=gpu:1
#SBATCH --mem=8000  # memory in Mb
#SBATCH --cpus-per-task=8
#SBATCH --time=0-12:00:00
#SBATCH -o output/mixture_compare_iw_%A.txt  # send stdout to outfile
#SBATCH -e output/mixture_compare_iw_error_%A.txt  # send stderr to errfile
#SBATCH --partition=apollo

source ~/.bashrc
conda activate deconv

python experiments/flows/mixture_compare.py -k 10 -e 300 -i -l 0.0001 -m 128 results/flows/iw_8/mixture_compare_10_iw_${SLURM_JOBID}
python experiments/flows/mixture_compare.py -k 25 -e 300 -i -l 0.0001 -m 128 results/flows/iw_8/mixture_compare_25_iw_${SLURM_JOBID}
python experiments/flows/mixture_compare.py -k 50 -e 300 -i -l 0.0001 -m 128 results/flows/iw_8/mixture_compare_50_iw_${SLURM_JOBID}






