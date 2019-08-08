""" This script will analyze a collection of models by saving only their best
    performances to a csv file. """

import pandas as pd
import numpy as np
from glob import glob
import os
import multiprocessing
from tqdm import tqdm


def get_title(filename):
    """
     >>> get_title("logs/0613/0613-q1-0000.train")
     '0613-q1-0000'
     """
    return os.path.splitext(os.path.basename(filename))[0]


def get_df_from_file(f):
    try:
        df = pd.read_csv(f)
    except:
        return None
    df = df[df["is_end_of_epoch"]].reset_index()
    return df


def get_best_validation_train_row_from_df(df):
    cols = ["drmsd", "rmse", "rmsd", "combined"]
    best_crit = "rmsd"
    # Separate valid and training losses
    try:
        dfval = df[df["is_val"]]
    except KeyError:
        return None
    # Select best row based on criteria
    val_row = dfval[dfval[best_crit] == dfval[best_crit].min()][cols]
    val_row.columns = [str(c) + "-val" for c in val_row.columns]
    train_row = df.iloc[dfval[dfval[best_crit] == dfval[best_crit].min()].index -1][cols]
    train_row.columns = [str(c) + "-train" for c in train_row.columns]
    train_row.index = val_row.index
    best_row = pd.concat([train_row, val_row], axis=1)
    epoch_time = dfval.iloc[1].time - dfval.iloc[0].time
    best_row["epoch_time"] = epoch_time
    return best_row


def get_cmd_line_args_from_name(model_name, cmd_file):
    with open(cmd_file,'r') as f:
        for line in f:
            if model_name in line:
                return line.strip()


if __name__ == "__main__":
    files = glob("logs/0613*.train")


    def work(filename):
        try:
            mydf = get_df_from_file(filename)
            if mydf is None: return None
            row = get_best_validation_train_row_from_df(mydf)
            if row is None: return None
            row["name"] = get_title(filename)
            row["cmd_line"] = get_cmd_line_args_from_name(row["name"].tolist()[0], "cluster/190613.txt")
            return row
        except:
            return None

    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        results = list(tqdm(p.imap(work, files), total=len(files)))

    final_df = pd.concat(results).sort_values("drmsd-val")
    final_df.to_csv("all_summary.csv", index=False)
