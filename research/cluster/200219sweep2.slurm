#!/bin/bash
#SBATCH --job-name=new-sweep
#SBATCH --nodes=1
#SBATCH --gres=gpus:1
#SBATCH -C "C6"
#SBATCH --cpus-per-gpu=16
#SBATCH --time=14-00:00:00
#SBATCH --partition=dept_gpu
#SBATCH --output="../research/cluster/slurm/200219slurm-%A_%a.out"


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
wandb agent --count 1 koes-group/protein-transformer/ji5vzfek

exit 0
