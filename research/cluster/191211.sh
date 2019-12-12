#!/bin/bash 

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                    Slurm Construction Section
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#job name 
#SBATCH --job name=pt-sweep

#job time
#SBATCH --time=28:00:00:00

#job memory
#SBATCH --mem=10g

#partition (queue) declaration
#SBATCH --partition=dept_gpu

#number of requested nodes
#SBATCH --nodes=1

#number of requested gpus
#SBATCH --gres=gpu:1

#requested array dimension
#SBATCH --array=1-4

#number of tasks
#SBATCH --ntasks=1

#number of requested cores
#SBATCH --ntasks-per-node=2

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                    User Construction Section
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#current (working) directory
work_dir=$(pwd)

#username
user=$(whoami)

#directory name where job will be run (on compute node)
job_dir="${user}_${SLURM_JOB_ID}.dcb.private.net"

#create directory on /scr folder of compute node
mkdir /scr/$job_dir

#change to the newly created directory
cd /scr/$job_dir

#copy the submit file (and all other related fields/directories)
rsync -a ${work_dir}/* .

#put date and time of starting job in a file
date>date-$SLURM_ARRAY_TASK_ID.txt

#runs stress-ng (to put stress on node) for 120 seconds
#stress-ng --cpu$SLURM_TASK_PER_NODE --timeout 120s --metrics-brief>stress-ng -$SLURM_ARRAY_TASK_ID.log$SLURM_ARRAY_TASK_ID
nvidia-smi

#put hostname of compute node in a file
hostname>hostname-$SLURM_ARRAY_TASK_ID.txt

#append date and time of finished job in a file
date>>date-$SLURM_ARRAY_TASK_ID.txt

#copy back all the files/directories back to the Slurm head node
rsync -ra *.log *.txt ${work_dir}