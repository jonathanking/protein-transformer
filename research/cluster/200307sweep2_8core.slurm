#!/bin/bash
#SBATCH --job-name=friendly-sweep
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -C "C6"
#SBATCH --ntasks-per-node=8
#SBATCH --time=14-00:00:00
#SBATCH --partition=dept_gpu
#SBATCH --output="../research/cluster/slurm/200307slurm-%A_%a.out"


############################
##       Environment      ##
############################
eval "$(conda shell.bash hook)"
conda activate pytorch_conda_200307
module load cuda/10.1


############################
##  Move to Scratch Drive ##
############################
mkdir -p /scr/jok120
cd /scr/jok120
if ! [[ -d protein-transformer ]]
then
    echo "Cloning repo."
    git clone git@github.com:jonathanking/protein-transformer.git
    rsync /net/pulsar/home/koes/jok120/protein-transformer/data/proteinnet/casp12_200218_30.pt protein-transformer/data/proteinnet
    cd protein-transformer/protein_transformer
else
    cd protein-transformer/protein_transformer
    git pull
fi
echo "Updated repo."

############################
##     Array Job Exec.    ##
############################
wandb agent --count 1 koes-group/protein-transformer/ji5vzfek
echo "Completed wandb agent."


############################
##     Copy Files Back    ##
############################
rsync -avzh --exclude "*.gltf" --exclude "*.png" wandb /net/pulsar/home/koes/jok120/protein-transformer/protein_transformer/wandb
echo "Completed rsync."

exit 0
