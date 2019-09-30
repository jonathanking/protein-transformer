"""
Plots a model's .train file.
"""
import argparse
from datetime import date
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
from scipy.signal import savgol_filter

def get_model_name(filename):
    """
     >>> get_model_name("logs/0613/0613-q1-0000.train")
     '0613-q1-0000'
     """
    return os.path.splitext(os.path.basename(filename))[0]


def get_df_from_file(f):
    """ Loads a CSV formatted file as a pandas dataframe."""
    df = pd.read_csv(f)
    df = df.reset_index()
    return df


def smooth(ys):
    """ Uses Savgol filter method to produce a smoothed version of ys. """
    smooth_size = min(len(ys) // 7, 301)
    smoothed = savgol_filter(ys, smooth_size, 2)
    return smoothed

def title_to_fn(title):
    """ Given a plot title, returns the same title, suitable as a filename. Removes punctuation, etc."""
    title = title.replace(" = [", "_").replace(", )", "_").replace(", ", "_")  # Remove time notation
    title = title[title.find('\n')+1:]
    replace_chars = " ,=[]().$"
    for rc in replace_chars:
        title = title.replace(rc, "_")
    title = title.lower()
    return title


def plot(dftrain, dfval, metric, title, outpath, smoothing=False, skip=None, include_val=False):
    """ Simple plotting function. Given train/val data and options, saves fig outpath. """
    dftrain_new = dftrain.iloc[skip:]

    sns.lineplot(x=dftrain_new.index, y=metric, data=dftrain_new, label=metric, alpha=0.6)
    if smoothing:
        sns.lineplot(x=dftrain_new.index, y=smooth(dftrain_new[metric]), color="lightblue")
    if include_val:
        sns.lineplot(x=dfval.index, y=metric, data=dfval, label=metric+"-val")

    plt.ylabel("Loss Value")
    plt.xlabel("Iteration Number")
    plt.title(title)
    plt.savefig(outpath + title_to_fn(title) + ".png")
    plt.figure()


def main():
    # Setup
    df = get_df_from_file(args.train_file)
    dftrain = df[df["is_val"] != True]
    dfval = df[df["is_val"] & df["is_end_of_epoch"]]
    model_name = get_model_name(args.train_file)
    today = date.today().strftime("%y%m%d")
    outpath = f"../research/analysis/{today}/{model_name}/"
    os.makedirs(outpath, exist_ok=True)
    smoothing = not args.no_smoothing

    # Plots are defined by a list of metrics
    metrics = ["drmsd", "rmse", "combined", "ln_drmsd"]
    titles = [m + " Loss" for m in metrics]

    # We then create 5-tuples to later feed into the plot fn. These contain:
    #    -- (metric, plt_title, smoothing, include_val, skip)
    all_plots = [(m, t, smoothing, args.val, None) for (m, t) in zip(metrics, titles)]

    # This plots are the same as normal_plots, but they start at iteration args.skip_first.
    if args.skip_first:
        skip_plots = []
        for (metric, plt_title, _, _, _) in all_plots:
            skip_plots.append((metric, plt_title + f", t = [{args.skip_first}, )", smoothing, args.val, args.skip_first))
        all_plots += skip_plots

    # Add model name to all plot titles
    all_plots = [(metric, model_name + "\n" + plt_title, s, v, skip) for (metric, plt_title, s, v, skip) in all_plots]

    # Make all plots
    for (metric, plt_title, sm, val, skip) in all_plots:
        plot(dftrain, dfval, metric, plt_title, outpath, smoothing=sm, include_val=val, skip=skip)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Makes plots for a given model file.")
    parser.add_argument('train_file', type=str, help="A model's .train file.")
    parser.add_argument('-v', '--val', action="store_true", help="Plot validation as well as train.")
    parser.add_argument('-sf', '--skip_first', type=int, help="Number of iterations at start to skip.")
    parser.add_argument('--no_smoothing', action="store_true", help="Turn off plot smoothing.")
    args = parser.parse_args()
    main()
