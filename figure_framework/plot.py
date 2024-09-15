import argparse
import io
import logging
import os
import pprint
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from scipy.stats import t

import config
from conv_gan.utils import logger

pp = pprint.PrettyPrinter(indent=4)
log = logging.getLogger()

from conv_gan.models.dcgan import DCGan
from conv_gan.utils.parser import load_config_file

HALF_WIDTH = 252.0 * 0.0352778
FULL_WIDTH = 516.92913 * 0.0352778
DLOSS_COL = "D Loss"
ACTUAL_POINTS = "Actual"
GENERATED_POINTS = "Generated"
HEIGHT = 2.5
RED = 'tab:red'
BLUE = 'tab:blue'
TITLE = "Comparison of DCGAN and Noise-TrajGAN Models [{DATASET}]"
UNIT_MAP = {
    "Hausdorff Distance": "HD [1]",
    "Total Travelled Distance": "WD(TTD) [1/1000]",
    "Sliced Wasserstein Distance": "SWD [1]",
    "Time Reversal Ratio": "TRR [%]",
    "D Loss": "D Loss [1]",
    "HD": "HD [1]",
    "TTD": "WD(TTD) [1/1000]",
    "SWD": "SWD [1]",
    "TRR": "TRR [%]",
    "Discriminator Loss": "D Loss [1]"
}
font_size = 12
ticks_fontsize = font_size - 1
legend_font_size = font_size - 1
title_font_size = font_size + 1
settings = {
    'font.size': font_size,
    'figure.titlesize': title_font_size,
    'figure.autolayout': True,
    'legend.fontsize': legend_font_size,
    'axes.titlesize': font_size,
    'axes.labelsize': font_size - 1,
    'ytick.labelsize': ticks_fontsize,
    'xtick.labelsize': ticks_fontsize,
    'hatch.linewidth': 0.8,
    'xtick.minor.pad': 1,
    'axes.labelpad': 3,
    'legend.framealpha': 1,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    'legend.handletextpad': 0.2,
    'legend.columnspacing': 0.8,
    'figure.dpi': 1000,
    'legend.facecolor': 'white',
    'lines.linewidth': 1.5,
    'errorbar.capsize': 3,
    'lines.markeredgewidth': 0.7,
    'lines.markersize': 6,
}
plt.rcParams.update(settings)


def save_fig(fig, filename):
    fig.tight_layout()
    # Create the directory if it doesn't exist
    if not os.path.exists(config.PLOT_DIR):
        os.makedirs(config.PLOT_DIR, exist_ok=True)
    if not filename.endswith(".png"):
        filename += ".png"
    filepath = os.path.join(config.PLOT_DIR, filename)
    fig.savefig(filepath, bbox_inches='tight')
    print(f"Saved {filepath}")


def calculate_t_confidence_interval(data, confidence=0.80):
    # Ensure data is a numpy array for easier mathematical operations
    data = np.array(data)
    # Calculate the mean of the data
    mean = np.mean(data)
    # Calculate the standard deviation of the data
    std_dev = np.std(data, ddof=1)  # ddof=1 provides sample standard deviation
    # Find the number of observations
    n = len(data)
    # Calculate the degrees of freedom
    df = n - 1
    # Calculate the critical t-value
    t_critical = t.ppf((1 + confidence) / 2, df)
    # Calculate the margin of error
    margin_of_error = t_critical * (std_dev / np.sqrt(n))
    # Calculate the confidence interval
    return mean, margin_of_error


def plot_line(axis: plt.Axes, x: Iterable[float], data: Iterable[float], label: str = None, color=None,
              dataset="geolife", marker=None):
    """
    Plot a line with error-bars at each point.
    :param axis: The Axes object to plot onto
    :param x: The x coordinates len(x) == len(data)
    :param data: List of y values to compute the mean and confidence interval from
    :param label: Line label for legend
    :param color: Line color
    :param dataset: The dataset name
    :param marker: The marker to use
    :return:
    """
    y = []
    for d in data:
        d = np.array(d)
        # Calculate the mean of the data
        mean = np.mean(d)
        y.append(mean)
    final_mean = y[-1]
    # Basically if we are looking at 
    if label in ['Total Travelled Distance', "TTD"]:
        y = [val / 1000 for val in y]
    elif label in ['Time Reversal Ratio', 'TRR']:
        y = [val * 100 for val in y]

    # if dp and dataset == "geolife":
    #     x = [val / 10 for val in x]

    if dataset == "geolife":
        x = [int(val / 90) for val in x]
    elif dataset == "fs":
        x = [int(val / 39) for val in x]
    axis.plot(
        x[:11],
        y[:11],
        color=color,
        marker=marker,
    )
    return {label: final_mean}


def get_eval_photos(plot_examples=True, gpu=0):

    # Run the model with eval
    # Get the real, generated and point cloud photos
    opt = {
        "config": "configs/cnngan_geolife_gan.json",
        "dataset": "geolife",
        "gpu": 0
    }

    # Eval and get trajectories before and after training + Get the point clouds
    opt, dcgan = load_dcgan(opt)
    real, generated_before, (_, _) = dcgan.eval_model(0)
    epochs = opt["epochs"]
    # epochs = 100
    real1, generated_after, (actual_clouds, gen_clouds) = dcgan.eval_model(epochs)

    opt = {
        "config": "configs/dp-cnngan_geolife_iwgan.json",
        "dataset": "geolife",
        "gpu": 0
    }
    opt, dcgan = load_dcgan(opt)

    # Get the point clouds for dp (Can be improved by seperating these functions)
    _, _, (actual_cloudsdp, gen_cloudsdp) = dcgan.eval_model(opt["epochs"])

    if plot_examples:
        # Plot all 10 progress plots
        for i in range(len(real)):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(HALF_WIDTH, HEIGHT))
            ax1.imshow(plotly_to_PIL(real[i]))
            ax1.tick_params(left=False, right=False, labelleft=False,
                            labelbottom=False, bottom=False)
            ax1.set_title(f'Actual Trajectory', fontsize=title_font_size)
            ax2.imshow(plotly_to_PIL(generated_before[i]))
            ax2.tick_params(left=False, right=False, labelleft=False,
                            labelbottom=False, bottom=False)
            ax2.set_title(f'Start of Training', fontsize=title_font_size)
            ax3.imshow(plotly_to_PIL(generated_after[i]))
            ax3.tick_params(left=False, right=False, labelleft=False,
                            labelbottom=False, bottom=False)
            ax3.set_title(f'End of Training', fontsize=title_font_size)
            save_fig(fig, f"progress{i}")

    # Plot the Point clouds
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(HALF_WIDTH, HEIGHT))
    ax1.set_title("Point Distribution DCGAN", fontsize=title_font_size)
    ax1.scatter(actual_clouds[0], actual_clouds[1], s=1, label=ACTUAL_POINTS, c="blue")
    ax1.scatter(gen_clouds[0], gen_clouds[1], s=1, label="DCGAN", c="orange")
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    # Use larger symbols for the legend, position it in upper left
    ax1.legend(loc='upper left', markerscale=5)
    ax2.set_title("Point Distribution DP-DCGAN", fontsize=title_font_size)
    ax2.scatter(actual_cloudsdp[0], actual_cloudsdp[1], label=ACTUAL_POINTS, s=1, c="blue")
    ax2.scatter(gen_cloudsdp[0], gen_cloudsdp[1], label="DP-DCGAN", s=1, c="orange")
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.legend(loc='upper left', markerscale=5)
    save_fig(fig, "point_clouds")


def load_dcgan(opt, eval_mode=True):
    opt = load_config_file(opt)
    if opt["dp"]:
        print(f"Using DP CNNGAN {opt['dp']}")
    else:
        print("Using Standard CNNGAN")
    # Load dataset based on dataset_name
    print(f"Loading dataset for CNN-GAN: {opt['dataset']}")
    data = pd.read_csv("data/geolife/restricted_geolife.csv")
    trajectories = [traj.values.tolist()[:144] for tid, traj in data.groupby('tid')]
    opt["load_params"] = 110
    dcgan = DCGan(
        opt,
        mainData=trajectories,
        testData=trajectories,
        fold=0,
        dp=opt["dp"],
        privacy_accountant='prv',
        gpu=opt["gpu"],
        max_grad_norm=opt["max_grad_norm"],
        eval_mode=eval_mode,
        epsilon=opt["epsilon"] if opt["dp"] else None,
    )

    return opt, dcgan


def plotly_to_PIL(fig):
    # convert Plotly fig to  an array
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    return Image.open(buf)


def create_progress_visualisation():
    # Run plot eval with 0 params 
    # Run plot eval with max params
    # Save 3 photos (real, before, after) to a file
    pass


def plot_clouds():
    # Run plot eval
    pass


def plot_trends(folder):
    # Define the 4 figures of the subplots.
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(HALF_WIDTH, HEIGHT))

    filenames = [os.path.join(folder, filename) for filename in os.listdir(folder) if "metrics" in filename]

    dataframes = [pd.read_csv(filename) for filename in filenames]
    metrics = dataframes[0].columns
    batches = dataframes[0]["Batches"].values
    saved_data = []

    # Get all the metrics data into horizontally stacked arrays
    for metric in metrics[1:]:
        # Extract this column from each dataframe and stack them horizontally
        stacked_column_data = np.hstack([df[metric].values.reshape(-1, 1) for df in dataframes])
        saved_data.append(stacked_column_data)

    # Plot HD against SWD
    plot_twins(ax1, batches, batches, saved_data[0], saved_data[2], ["HD", "SWD"])
    ax1.set_title(f'DCGAN Geolife HD and SWD Trend')

    # Get all the loss dataframes from the "folds" files
    loss_frames = [pd.read_csv(folder + filename) for filename in os.listdir(folder) if filename.startswith("fold")]

    # Stack the Loss
    stacked_DLOSS = np.hstack([df[DLOSS_COL].values.reshape(-1, 1) for df in loss_frames])

    # Downsample the loss
    stacked_DLOSS = stacked_DLOSS[::100]
    loss_batches = loss_frames[0]["Batches"].values

    # Downsample the loss batches
    loss_batches = loss_batches[::100]

    # Plot loss against the TTD
    plot_twins(ax3, loss_batches, batches, stacked_DLOSS, saved_data[1], ["D Loss", "TTD"])
    ax3.set_title(f'DCGAN Geolife Loss and TTD Trend')
    labels = get_labels([folder])
    filename = labels[0] + "_SWD_AND_HD_trends"
    save_fig(fig, filename)


def plot_twins(ax, y1, y2, x1, x2, labels):
    plot_line(ax, y1, x1, label=labels[0], color=RED)

    ax.set_xlabel('Epochs')

    ax.set_ylabel(labels[0], color=RED)
    ax.tick_params(axis='y', labelcolor=RED)
    # ax.set_ylim(bottom=0)
    ax2 = ax.twinx()
    ax2.set_ylabel(labels[1], color=BLUE)
    plot_line(ax2, y2, x2, label=labels[1], color=BLUE)
    ax2.tick_params(axis='y', labelcolor=BLUE)
    # ax2.set_ylim(bottom=0)
    add_axis_labels_to_plot([ax, ax2], labels)


def plot_results_from_folders(ntg_folder=None, cnngan_folder=None, dp_ntg_folder=None, dp_cnngan_folder=None,
                              labels=None, output_file=None, input_dirs=None):
    # Make sure either input_dirs or the individual folders are provided
    if input_dirs is None and (ntg_folder is None or cnngan_folder is None):
        raise ValueError("Either input_dirs or the individual folders must be provided!")
    if input_dirs is not None:
        # Make sure sufficient labels are provided
        if len(input_dirs) != len(labels):
            raise ValueError("Number of labels must match the number of directories!")
        folders = input_dirs
    else:
        # Select all folders that are not none
        folders = [folder for folder in [cnngan_folder, ntg_folder, dp_cnngan_folder, dp_ntg_folder] if
                   folder is not None]
    # Define the 4 figures of the subplots.
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(FULL_WIDTH, HEIGHT))
    plots = [ax1, ax2, ax3, ax4]
    dataset = "geolife" if "geolife" in folders[0] else "fs"
    results = {}
    metrics = []
    markers = ['o', 's', '^', 'D']  # circle, square, triangle up, diamond
    for j, folder in enumerate(folders):
        log.info("Plotting: " + folder)
        dataframes = [pd.read_csv(os.path.join(folder, filename)) for filename in sorted(os.listdir(folder)) if
                      "metrics" in filename]

        # The last measurement (if there are multiple) might not be completed yet
        # Therefore, if there is more than one measurement, we remove the last one if it is not complete
        if len(dataframes) > 1 and len(dataframes[-1]) < len(dataframes[0]):
            dataframes = dataframes[:-1]
            log.warning(f"Removed last measurement from {folder} as it is not complete")

        metrics = dataframes[0].columns[1:]
        batches = dataframes[0]["Batches"].values
        log.debug(f"Folder: {folder} - Num Batches: " + str(len(batches)))
        results[folder] = []
        metrics = [metrics[0], metrics[2], metrics[1], metrics[3]]
        for i, metric in enumerate(metrics):
            # Extract this column from each dataframe and stack them horizontally
            stacked_column_data = np.hstack([df[metric].values.reshape(-1, 1) for df in dataframes])
            # Store in the dictionary
            mean = plot_line(plots[i], batches, stacked_column_data, label=metric, dataset=dataset,
                             marker=markers[j % len(markers)])
            results[folder].append(mean)
    pprint.pprint(
        results
    )
    # Make sure the bottom value is 0 in the plots
    for plot in plots:
        plot.set_ylim(bottom=0)
    if labels is None:
        labels = get_labels(folders)
    else:
        assert len(labels) == len(folders), "Number of labels must match the number of folders"
    add_legend_super(fig, labels, plots)
    # change first geolife to Geolife and fs to FS-NYC
    dataset_name = "Geolife" if "geolife" in dataset else "FS-NYC"
    # Move title slightly up to make space for the legend
    fig.suptitle(TITLE.format(DATASET=dataset_name), y=1.01)
    add_axis_labels_to_plot(plots, metrics)
    filename = f"metrics_{dataset}" if output_file is None else output_file
    save_fig(fig, filename)


def get_labels(folders):
    labels = []
    for folder in folders:
        directory = folder.split("/")[-1]
        dir_split = directory.split("_")
        # print(folder)
        # Labels should be the model name
        labels.append(dir_split[0])
    return labels


def add_legend_super(fig, labels, axis):
    i = 0
    for line in axis[0].lines:
        line.set_label(labels[i])
        i += 1
    # Add a legend underneath the figure title with all labels horizontally displayed
    legend = fig.legend(labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 0.96))
    return legend


def add_axis_labels_to_plot(plots, metrics):
    for i in range(len(plots)):
        plots[i].set_xlabel("Epochs")
        plots[i].set_ylabel(UNIT_MAP[metrics[i]])


def plot_main_results(ntg_folder, cnngan_folder, dpntg_folder=None, dpcnngan_folder=None, labels=None, output_file=None,
                      input_dirs=None, **kwargs):
    plot_results_from_folders(ntg_folder=ntg_folder, cnngan_folder=cnngan_folder,
                              dp_ntg_folder=dpntg_folder, dp_cnngan_folder=dpcnngan_folder,
                              labels=labels, output_file=output_file, input_dirs=input_dirs)
    # plot_trends(cnngan_folder)
    # Do the following only for Geolife
    if cnngan_folder is not None and "geolife" in cnngan_folder.lower():
        get_eval_photos(False, gpu=kwargs["gpu"] if "gpu" in kwargs else 0)


def parse_args():
    parser = argparse.ArgumentParser(description='Argument Parser')
    # NTG and CNN-GAN
    parser.add_argument('-n', '--ntg', type=str, help='Path to NTG results folder')
    parser.add_argument('-c', '--cnngan', type=str, help='Path to CNNGAN results folder')
    # DP-NTG and DP-CNN-GAN
    parser.add_argument('-dn', '--dp-ntg', type=str, help='Path to DP-NTG results folder')
    parser.add_argument('-dc', '--dp-cnngan', type=str, help='Path to DP-CNN-GAN results folder')

    # Optionally, allow a list of directories as input
    parser.add_argument('-r', '--results', nargs='*', help='List of directories to plot',
                        default=None)

    # Allow a list of labels as input
    parser.add_argument('-l', '--labels', nargs='*', help='List of labels to use for the legend',
                        default=None)

    # Output file
    parser.add_argument('-o', '--output', type=str, help='Output file name', default=None)

    # GPU
    parser.add_argument('-g', '--gpu', type=int, help='GPU to use', default=0)

    return parser.parse_args()


if "__main__" == __name__:

    # Configure the logger
    log = logger.configure_root_loger(logging_level=logging.DEBUG, file=os.path.join(config.LOG_DIR, "plot.log"))

    args = parse_args()

    # Make sure the directories and labels are the same length if both are provided
    if args.results is not None or args.labels is not None:
        assert len(args.results) == len(args.labels), "Number of directories and labels must match, if provided!"

    plot_main_results(cnngan_folder=args.cnngan, ntg_folder=args.ntg,
                      dpntg_folder=args.dp_ntg, dpcnngan_folder=args.dp_cnngan,
                      labels=args.labels, output_file=args.output,
                      input_dirs=args.results, gpu=args.gpu)
