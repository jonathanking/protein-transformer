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
export PATH=/usr/local/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64/
#eval "$(cat ~/bin/conda_init.sh)"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate pytorch-build


############################
##     Array Job Exec.    ##
############################
cd $SLURM_SUBMIT_DIR
cmd="$(sed -n "${SLURM_ARRAY_TASK_ID}p" research/cluster/191211_test.txt)"
echo $cmd
eval $cmd

exit 0
