"""
This script will analyze a group of models. It will create loss plots and model predictions as requested.
Author: Jonathan King
July 10, 2019

predict.py usage:
usage: predict.py [-h] [-data DATA] [-dataset {train,valid,test,all}] [-n N]
                  [--pdb_dir PDB_DIR] [-bb] [--reconstruct] [--include_truth]
                  model_chkpt outdir

"""
import argparse
import torch
from .predict import load_model, get_data_loader, make_predictions, make_pdbs

def generate_pdbs(chkpt):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzes a group of models.")
    parser.add_argument('glob_pattern', type=str, help="Pattern used to select all relevant model checkpoints.")
    args = parser.parse_args()

