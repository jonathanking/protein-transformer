#!/bin/bash
#SBATCH --job-name=pt-sweep
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=10g
#SBATCH --time=28-00:00:00
#SBATCH --partition=dept_gpu


############################
##       Environment      ##
############################
cd $SLURM_SUBMIT_DIR
export PATH=/usr/local/bin:$PATH
export PATH=/opt/anaconda3/bin:$PATH
. $CONDA_ROOT/etc/profile.d/conda.sh
export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64/
conda activate pytorch-build

############################
##     Array Job Exec.    ##
############################
cmd="$(sed -n "${SLURM_ARRAY_TASK_ID}p" research/cluster/191211_test.txt)"
echo $cmd
eval $cmd


############################
##          Exit          ##
############################
exit 0
