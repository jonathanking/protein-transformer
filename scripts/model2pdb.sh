#!/usr/bin/env bash


### This script takes the name of a trained model, loads it from the "checkpoints" directory


if [ $# = 2 ]; then
   MODEL=$1  # trighelix01
   DATASET=$2
else
   echo "Please provide a model to load and predict from, as well as a dataset {train,valid,test} to use."
   exit 0
fi

DIR=$MODEL

if [ ! -d "$DIR" ]; then
    mkdir  "$DIR"
fi

PREDFILE="coords/$(basename "$MODEL")_${DATASET}.coord"

python load_predict.py checkpoints/${MODEL}.chkpt $PREDFILE $DATASET
python coords2pdb.py $PREDFILE $DIR

gunzip -f $DIR/*.gz

pymol $DIR/*.pdb
