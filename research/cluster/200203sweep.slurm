#!/bin/bash
#SBATCH --job-name=custom-sweep
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -C "C6"
#SBATCH --cpus-per-gpu=16
#SBATCH --time=14-00:00:00
#SBATCH --partition=dept_gpu
#SBATCH --output="../research/cluster/slurm/200202slurm-%A_%a.out"


############################
##       Environment      ##
############################
eval "$(conda shell.bash hook)"
conda activate psp
module load cuda/10.1


############################
##     Array Job Exec.    ##
############################
cd $SLURM_SUBMIT_DIR
cmd="$(sed -n "${SLURM_ARRAY_TASK_ID}p" ../research/cluster/200203sweep.txt)"
echo $cmd
eval $cmd

exit 0
