#!/bin/bash
#SBATCH --job-name=pt-array
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -C "C6&M12"
#SBATCH --cpus-per-gpu=4
#SBATCH --time=28-00:00:00
#SBATCH --partition=dept_gpu
#SBATCH --output="../research/cluster/slurm/slurm-%A_%a.out"


############################
##       Environment      ##
############################
eval "$(conda shell.bash hook)"
conda activate pytorch_c6m12_cuda101
module load cuda/10.1
echo $(which python)
echo "${SLURM_ARRAY_TASK_ID}"

############################
##     Array Job Exec.    ##
############################
cd $SLURM_SUBMIT_DIR
cmd="$(sed -n "${SLURM_ARRAY_TASK_ID}p" ../research/cluster/200127_pt_array.txt)"
echo $cmd
eval $cmd

exit 0
