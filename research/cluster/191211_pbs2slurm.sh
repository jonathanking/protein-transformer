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
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/net/mahler/opt/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
   eval "$__conda_setup"
else
    if [ -f "/net/mahler/opt/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/net/mahler/opt/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/net/mahler/opt/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<


cd $SLURM_SUBMIT_DIR
export PATH=/usr/local/bin:$PATH
export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64/
conda activate pytorch-build
which python

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
