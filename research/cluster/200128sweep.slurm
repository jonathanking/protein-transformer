#!/bin/bash
#SBATCH --job-name=protein-sweep
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -C "C6"
#SBATCH --cpus-per-gpu=4
#SBATCH --time=1-00:00:00
#SBATCH --partition=dept_gpu
#SBATCH --output="research/cluster/slurm/200128sweep/slurm-%A_%a.out"


############################
##       Environment      ##
############################
conda activate pytorch_c6m12_cuda101
module load cuda/10.1


############################
##     Array Job Exec.    ##
############################
cd $SLURM_SUBMIT_DIR
wandb agent --count 1 koes-group/protein-transformer/73aevuo9

exit 0
