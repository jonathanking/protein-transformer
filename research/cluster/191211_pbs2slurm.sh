#!/bin/bash
#SBATCH --job-name=pt-sweep
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10g
#SBATCH --time=28:00:00:00
#SBATCH --partition=dept_gpu


############################
##       Environment      ##
############################
cd $SLURM_SUBMIT_DIR
export PATH=/usr/local/bin:$PATH
export PATH=/opt/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64/
conda activate pytorch-build

############################
##     Array Job Exec.    ##
############################
cmd="/net/pulsar/home/koes/jok120/.conda/envs/pytorch-build/bin/$(sed -n "${SLURM_ARRAY_TASK_ID}p" ../research/cluster/191206.txt)"
echo $cmd
eval $cmd


############################
##          Exit          ##
############################
exit 0
