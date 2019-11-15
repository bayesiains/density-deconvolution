#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH --gres=gpu:1
#SBATCH --mem=8000  # memory in Mb
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH -o output/sgd-%A_%a.txt  # send stdout to outfile
#SBATCH -e output/sgd_error-%A_%a.txt  # send stderr to errfile
#SBATCH --partition=apollo

source ~/.bashrc
conda activate deconv

python deconv/experiments/gaia/variable_n/fit_gaia_sgd.py -c 512 -b 512 -e 10 -l 0.001 -w=0.001 -k 5 --lr-step 10 --lr-gamma 0.1 --train-limit 100000000 --use-cuda /disk/scratch/s0904254/gaia_source_shuffled.h5 results/variable_n/sgd_512_100_${SLURM_JOBID}
