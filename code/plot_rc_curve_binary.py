import numpy as np
import argparse

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as mcolors
import seaborn as sns
sns.set()
COLORS = list(mcolors.TABLEAU_COLORS)
N_COLORS = len(COLORS)

# === add abs path for import convenience
import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
from utils.utils import clear_terminal_output
from utils.rc_curve_utils import RC_curve, acc_coverage_curve


# GLOBAL VARs
PLOT_SYMBOL_DICT = {
    # plt line style definitions
    "max_sr": [0, "o", r"$SR_{max}$", "solid"], 
    "conf_margin": [4, "p", r"$RL_{conf-M}$", "solid"]
}


def read_data(root_dir, split="test_set", load_classifier_weight=False):
    raw_logits = np.load(os.path.join(root_dir, split, "pred_logits.npy"))
    labels = np.load(os.path.join(root_dir, split, "labels.npy"))
    last_layer_weights = None
    last_layer_bias = None
    if load_classifier_weight:
        weights_dir = os.path.join(root_dir, "last_layer_weights.npy")
        if os.path.exists(weights_dir):
            last_layer_weights = np.load(weights_dir)
            last_layer_bias = np.load(os.path.join(root_dir, "last_layer_bias.npy"))
    return raw_logits, labels, last_layer_weights, last_layer_bias


def plot_curve(
        covereage_list, risk_acc_list, fig_name,
        plot_symbol_dict, curve_name="risk-coverage"
    ):
    # if curve_name == "risk-coverage":
    # elif curve_name == "acc-coverage":

    # === Plot RC Curve ===
    plot_n_points = 30
    min_num_samples = -100
    save_path = fig_name
    line_width = 1
    markersize = 1
    alpha = 0.5

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    font_size = 19
    tick_size = 20

    y_min = 0
    y_max = 0
   
    method_name = "max_sr"
    x_plot, y_plot = covereage_list, risk_acc_list
    y_max, y_min = max(y_plot[0], y_max), min(np.amin(y_plot), y_min)
    # y_max, y_min = max(np.amax(y_plot), y_max), min(np.amin(y_plot), y_min)
    plot_settings = plot_symbol_dict[method_name]
    ax.plot(
        x_plot, y_plot,
        label=curve_name, lw=line_width, alpha=alpha,
        color=COLORS[plot_settings[0]], marker=plot_settings[1], ls=plot_settings[3], markersize=markersize
    )

    ax.legend(
        loc='lower left', bbox_to_anchor=(-0.25, 1, 1.25, 0.2), mode="expand", 
        borderaxespad=0,
        ncol=3, fancybox=True, shadow=False, fontsize=font_size, framealpha=0.3
    )
    ax.tick_params(axis='x', which='major', colors='black', labelsize=tick_size)
    ax.tick_params(axis='y', which='major', colors='black', labelsize=tick_size)
    ax.set_ylabel(r"%s" % curve_name.split("-")[0], fontsize=font_size)
    ax.set_xlabel(r"Coverage", fontsize=font_size)
    if curve_name == "risk-coverage":
        ax.set_ylim([y_min-0.05*y_max, 1.10*y_max])
        ax.set_xticks([0, 0.5, 1])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_xlim([-0.02, 1.05])
        ax.set_yticks([y_max/2, y_max])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def main(args):
    gamma = args.gamma
    # === Root dir to read collected data ===
    root_dir = args.root_dir
    exp_dir = args.exp_dir
    read_root_dir = os.path.join(root_dir, exp_dir)

    # === Root dir to save processed 
    save_root_dir = os.path.join("process_rc_data", exp_dir, "in-d")
    os.makedirs(save_root_dir, exist_ok=True)

    # ===  Load In-D collected data ===
    in_d_logits, in_d_labels, fc_weights, fc_bias = read_data(read_root_dir, split="test_set", load_classifier_weight=False)
    print("Check In-D shapes: ", in_d_logits.shape, in_d_labels.shape)
    
    save_rc_curve_root = os.path.join(save_root_dir, "rc_curves")
    os.makedirs(save_rc_curve_root, exist_ok=True)

    # ==== Process experiment data ====
    pred = np.where(in_d_logits > gamma, 1, 0)
    acc = np.mean(pred == in_d_labels) * 100
    print("Acc : %.04f" % acc)

    confidence_list = np.abs(gamma - in_d_logits)
    acc_list = np.where(pred==in_d_labels, 1, 0)
    residual_list = np.where(pred==in_d_labels, 0, 1)

    coverage_rc, risk_rc = RC_curve(residual_list, confidence_list)
    coverage_acc, risk_acc = acc_coverage_curve(acc_list, confidence_list)

    rc_fig_name = os.path.join(save_rc_curve_root, "risk-coverage-curve.png")
    plot_curve(coverage_rc, risk_rc, rc_fig_name, PLOT_SYMBOL_DICT)

    acc_fig_name = os.path.join(save_rc_curve_root, "acc-coverage-curve.png")
    plot_curve(coverage_acc, risk_acc, acc_fig_name, PLOT_SYMBOL_DICT, curve_name="acc-coverage")


    pass


if __name__ == "__main__":

    clear_terminal_output()
    print("This script takes into in-D logits and labels and process the RC curve.")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir", dest="root_dir", type=str,
        default=os.path.join(".", "raw_data_collection"),
        help="Root folder where the collected logit/label data are located."
    )
    parser.add_argument(
        "--exp_dir", dest="exp_dir", type=str,
        default="PE_net\\PE",
        help="Experiment subfolder where collected data are located."
    )
    parser.add_argument(
        "--gamma", dest="gamma", type=float,
        default=0.5,
        help="Binary cls decision boundary."
    )
    args = parser.parse_args()
    main(args)
    print("All task completed.")
