import numpy as np
import argparse

# === add abs path for import convenience
import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
from utils.utils import clear_terminal_output
from utils.rc_curve_utils import compute_recalls, calculate_score_acc, calculate_score_residual, \
    plot_rc_curve, plot_sample_percentage_coverage_curve, calculate_score_sample_cls, \
    plot_recall_coverage_curve


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


DATASET_NAMES = ["clean", "defocus_blur", "jpeg_compression", "gaussian_blur"]
def main(args):
    # === Root dir to read collected data ===
    root_dir = args.root_dir
    exp_dir = args.exp_dir
    read_root_dir = os.path.join(root_dir, exp_dir)

    # === Root dir to save processed 
    save_root_dir = os.path.join("process_rc_data", exp_dir, "joint_dataset")
    os.makedirs(save_root_dir, exist_ok=True)

    total_logits, total_labels = [], []
    for dataset_name in DATASET_NAMES:
        read_data_folder = os.path.join(read_root_dir, dataset_name)
        # ===  Load In-D collected data ===
        logits, labels, fc_weights, fc_bias = read_data(read_data_folder, split="test_set", load_classifier_weight=True)

        print("Check In-D shapes: ", logits.shape, labels.shape)
        acc = np.mean(np.argmax(logits, axis=1) == labels) * 100
        print(" %s Acc - %.04f" % (dataset_name, acc))

        total_logits.append(logits)
        total_labels.append(labels)
    
    total_logits = np.concatenate(total_logits, axis=0)
    total_labels = np.concatenate(total_labels, axis=0)
    
    # Compute recalls and balanced accuracy
    recall_dict = compute_recalls(total_logits, total_labels)
    print("Check recalls: ", recall_dict)
    
    save_rc_curve_root = os.path.join(save_root_dir, "rc_curves")
    os.makedirs(save_rc_curve_root, exist_ok=True)
    # # === Check RC and acc-coverage curve ===
    acc_scores, acc_list, acc_method_list = calculate_score_acc(total_logits, total_labels)
    acc_fig_name = os.path.join(save_rc_curve_root, "acc-coverage-curve.png")
    acc_coverage_dict, acc_dict = plot_rc_curve(
        acc_scores, acc_list, acc_fig_name, acc_method_list, PLOT_SYMBOL_DICT, curve_name="acc-coverage"
    )

    # # === Check RC and acc-coverage curve ===
    risk_scores, risk_list, risk_method_list = calculate_score_residual(total_logits, total_labels)
    risk_fig_name = os.path.join(save_rc_curve_root, "risk-coverage-curve.png")
    risk_coverage_dict, risk_dict = plot_rc_curve(
        risk_scores, risk_list, risk_fig_name, risk_method_list, PLOT_SYMBOL_DICT, curve_name="risk-coverage"
    )

    # # === Plot sample rejection dynamics ===
    scores_dict, method_names, labels_list = calculate_score_sample_cls(total_logits, total_labels)
    fig_path = os.path.join(save_root_dir, "class-wise-sample-rejection")
    os.makedirs(fig_path, exist_ok=True)
    plot_sample_percentage_coverage_curve(scores_dict, method_names, labels_list, fig_path)

    # # === Plot recall-coverage curve
    fig_path = os.path.join(save_root_dir, "recall-coverage")
    os.makedirs(fig_path, exist_ok=True)
    plot_recall_coverage_curve(total_logits, total_labels, fig_path)


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
        default="HAM\\focal",
        help="Experiment subfolder where collected data are located."
    )
    args = parser.parse_args()
    main(args)
    print("All task completed.")
