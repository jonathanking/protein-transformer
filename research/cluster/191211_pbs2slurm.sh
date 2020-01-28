#!/bin/bash
#SBATCH --job-name=pt-sweep
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=10g
#SBATCH --time=28-00:00:00
#SBATCH --partition=dept_gpu
#SBATCH --output="research/cluster/slurm/slurm-%A_%a.out"


############################
##       Environment      ##
############################
eval "$(conda shell.bash hook)"
conda activate pytorch-build


############################
##     Array Job Exec.    ##
############################
cd $SLURM_SUBMIT_DIR
cmd="$(sed -n "${SLURM_ARRAY_TASK_ID}p" research/cluster/191211_test.txt)"
echo $cmd
eval $cmd

exit 0
