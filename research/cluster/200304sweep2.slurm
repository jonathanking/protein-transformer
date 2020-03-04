#!/bin/bash
#SBATCH --job-name=friendly-sweep
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -C "C6"200219sweep2.slurm
#SBATCH --time=14-00:00:00
#SBATCH --partition=dept_gpu
#SBATCH --output="../research/cluster/slurm/200304slurm-%A_%a.out"


############################
##       Environment      ##
############################
eval "$(conda shell.bash hook)"
conda activate pytorch_mar20
module load cuda/10.2


############################
##  Move to Scratch Drive ##
############################
mkdir -p /scr/jok120
cd /scr/jok120
if ! [[ -d protein-transformer ]]
then
    git clone git@github.com:jonathanking/protein-transformer.git
    rsync /net/pulsar/home/koes/jok120/protein-transformer/data/proteinnet/casp12_200218_30.pt protein-transformer/data/proteinnet
fi
cd protein-transformer/protein_transformer


############################
##     Array Job Exec.    ##
############################
wandb agent --count 1 koes-group/protein-transformer/ji5vzfek


############################
##     Copy Files Back    ##
############################
rsync -avzh --exclude "*.gltf" --exclude "*.png" wandb /net/pulsar/home/koes/jok120/protein-transformer/protein_transformer/wandb


exit 0
