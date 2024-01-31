import numpy as np
import argparse

# === add abs path for import convenience
import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
from utils.utils import clear_terminal_output
from utils.rc_curve_utils import compute_recalls


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



def main(args):
    # === Root dir to read collected data ===
    root_dir = args.root_dir
    exp_dir = args.exp_dir
    read_root_dir = os.path.join(root_dir, exp_dir)

    # === Root dir to save processed 
    save_root_dir = os.path.join("process_rc_data", exp_dir, "in-d")

    # ===  Load In-D collected data ===
    in_d_logits, in_d_labels, fc_weights, fc_bias = read_data(read_root_dir, split="test_set", load_classifier_weight=True)
    print("Check In-D shapes: ", in_d_logits.shape, in_d_labels.shape)

    print("Check In-D shapes: ", in_d_logits.shape, in_d_labels.shape)
    acc = np.mean(np.argmax(in_d_logits, axis=1) == in_d_labels) * 100
    print("Acc - %.04f" % acc)
    
    # Compute recalls and balanced accuracy
    recall_dict = compute_recalls(in_d_logits, in_d_labels)
    print("Check recalls: ", recall_dict)

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
        default="HAM-res50",
        help="Experiment subfolder where collected data are located."
    )
    args = parser.parse_args()
    main(args)
    print("All task completed.")
