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
conda activate pytorch_conda_200307
module load cuda/10.1


############################
##     Array Job Exec.    ##
############################
cd $SLURM_SUBMIT_DIR
cmd="$(sed -n "${SLURM_ARRAY_TASK_ID}p" research/cluster/200330.txt)"
echo $cmd
eval $cmd

exit 0
