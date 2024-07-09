import numpy as np
import os
import argparse
from sklearn.metrics import balanced_accuracy_score
from sklearn import metrics
from collections import Counter

# === add abs path for import convenience
import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
from utils.utils import clear_terminal_output
from utils.rc_curve_utils import compute_recalls, calculate_score_acc, calculate_score_residual, \
    plot_rc_curve, plot_sample_percentage_coverage_curve, calculate_score_sample_cls, \
    plot_recall_coverage_curve
    
from utils.metrics import aurc_eaurc, calc_aurc_eaurc, calc_fpr_aupr
    
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



def SR_sorting(labels, logits):
    '''
    sort based on softmax response
    '''
    n_samples = logits.shape[0]
    index = list(range(n_samples))
    
    softmax_max = np.max(logits, 1)
    sort_values = sorted(zip(softmax_max[:], index, labels), key=lambda x:x[0], reverse=True)
    sort_softmax_max, sort_index, sorted_labels = zip(*sort_values)
    sorted_labels = np.asarray(sorted_labels)
    sorted_logits = logits[sort_index, :]
    return sorted_labels, sorted_logits

def Confidence_sorting(labels, logits):
    n_samples = logits.shape[0]
    ## re index the column
    sorted_logits = np.sort(logits, axis=1)[:, ::-1]
    geo_score = sorted_logits[:, 0] - sorted_logits[:, 1]
    
    index = list(range(n_samples))
    sort_values = sorted(zip(geo_score, index, labels), key=lambda x:x[0], reverse=True)
    sort_geo_score, sort_index, sorted_labels = zip(*sort_values)
    sorted_labels = np.asarray(sorted_labels)
    sorted_logits = logits[sort_index, :]
    return sorted_labels, sorted_logits

def Margin_sorting(labels, logits, fc_weights):
    n_samples = logits.shape[0]
    ## re index the column
    fc_norm = np.linalg.norm(fc_weights, axis=1) ** 2
    for i in range(logits.shape[1]):
        logits[:, i] /= fc_norm[i]
    
    sorted_logits = np.sort(logits, axis=1)[:, ::-1]

    geo_score = sorted_logits[:, 0] - sorted_logits[:, 1]
    
    index = list(range(n_samples))
    sort_values = sorted(zip(geo_score, index, labels), key=lambda x:x[0], reverse=True)
    sort_geo_score, sort_index, sorted_labels = zip(*sort_values)
    sorted_labels = np.asarray(sorted_labels)
    sorted_logits = logits[sort_index, :]
    return sorted_labels, sorted_logits
    

def my_auc(vals):
    auc = 0
    for val_i in vals:
        auc += val_i * (1 / len(vals))
    return auc

def AUBAC_helper(sorted_labels, sorted_logits):
    n_samples = sorted_labels.shape[0]
    
    ## double check the balance accuracy
    ba = balanced_accuracy_score(sorted_labels, np.argmax(sorted_logits, axis=1))
    
    ## get a list of balance accuracy with incremental rejection, from low to high (reverse order)
    bas = [ba]
    for i in range(n_samples-1):
        tmp_labels = sorted_labels[:-(i+1)]
        tmp_logits = np.argmax(sorted_logits, axis=1)[:-(i+1)]
        tmp_ba = balanced_accuracy_score(tmp_labels, tmp_logits)
        bas.append(tmp_ba)
    bas.append(0)
    
    
    ## assume optimal classifer
    ## get how many mis-classified
    correct = np.sum(np.argmax(sorted_logits, axis=1) == sorted_labels)
    incorrect_ids = np.argmax(sorted_logits, axis=1) != sorted_labels
    incorrect_labels = sorted_labels[incorrect_ids]
    
    ## get the counter map for the total sample and incorrect counts to get BA
    incorrect_count = Counter(incorrect_labels)
    total_count = Counter(sorted_labels)
    
    ## check the best possible rejection
    n_incorrect = n_samples - correct
    bas2 = [ba]
    for i in range(n_samples-1): ## for every drop instance
        if i >= n_incorrect:
            bas2.append(1)
            continue
        
        ## find the best rejection sample
        best_BA = -1
        droped_class_idx = -1
        for drop_i in incorrect_count.keys():
            if incorrect_count[drop_i] == 0:
                continue
            ## calcualte ba for once the instance was dropped
            tmp_ba = []
            for class_i in total_count.keys():
                if class_i != drop_i:
                    acc = (total_count[class_i] - incorrect_count[class_i]) / total_count[class_i]
                else:
                    acc = ((total_count[class_i] - 1) - (incorrect_count[class_i] - 1)) / (total_count[class_i]-1)
                tmp_ba.append(acc)
            if best_BA < np.mean(tmp_ba):
                best_BA = np.mean(tmp_ba)
                droped_class_idx = drop_i
        
        ## update the count_map
        total_count[droped_class_idx] -= 1
        incorrect_count[droped_class_idx] -= 1
        bas2.append(best_BA)
    bas2.append(0)
        
    
    ## double check if all the count map is cleared, turn on for debug, should be all zeros
    # print(incorrect_count)
    
    ## reverse the list to the the ba-coverage correspondance: coverage 0% -> 100%
    bas = bas[::-1]
    bas2 = bas2[::-1]
    
    # coverage = [i/n_samples for i in range(n_samples+1)]    
    # aubac = metrics.auc(coverage, bas)
    # aubac_o = metrics.auc(coverage, bas2)
    aubac = my_auc(bas)
    aubac_o = my_auc(bas2)
    return aubac, aubac_o - aubac, ba

def AUBAC(logits, labels, SC_criterion="SR", **args):
    logits = np.array(logits)
    ## get balance accuracy
    ba = balanced_accuracy_score(labels, np.argmax(logits, axis=1))
    if SC_criterion == "SR":
        sorted_labels, sorted_logits = SR_sorting(labels, logits)
    elif SC_criterion == "Confidence_margin":
        sorted_labels, sorted_logits = Confidence_sorting(labels, logits)
    elif SC_criterion == "Geo_margin":
        sorted_labels, sorted_logits = Margin_sorting(labels, logits, args['last_fc_weights'])
    else:
        exit(f"{SC_criterion} is currently not supported")
    
    aubac, eaubac, ba =  AUBAC_helper(sorted_labels, sorted_logits)
    return aubac, eaubac, ba
     
def AURC_helper(sorted_labels, sorted_logits):
    n_samples = sorted_labels.shape[0]
    acc = np.mean(sorted_labels==np.argmax(sorted_logits, axis=1))
    risks = [1-acc]
    for i in range(n_samples-1):
        tmp_labels = sorted_labels[:-(i+1)]
        tmp_logits = np.argmax(sorted_logits, axis=1)[:-(i+1)]
        
        tmp_acc = np.mean(tmp_labels == tmp_logits)
        risks.append(1-tmp_acc)
    risks.append(0)
    
    risks2 = [1-acc]
    n_incorrect = n_samples - acc * n_samples
    for i in range(n_samples-1):
        if i >= n_incorrect:
            risks2.append(0)
            continue
        risks2.append((n_incorrect-i)/(n_samples-i))    
    risks2.append(0)
        
        
    risks = risks[::-1]
    risks2 = risks2[::-1]
        
    aurc = my_auc(risks)
    aurc_o = my_auc(risks2)
    
    # coverage = [i/n_samples for i in range(n_samples+1)]
    # aurc = metrics.auc(coverage, risks)
    # aurc_o = metrics.auc(coverage, risks2)
    return aurc, aurc-aurc_o, acc

def AURC(logits, labels, SC_criterion="SR", **args):
    logits = np.array(logits)
    acc = np.mean(labels==np.argmax(logits, axis=1))
    ## sort the labels
    if SC_criterion == "SR":
        sorted_labels, sorted_logits = SR_sorting(labels, logits)
    elif SC_criterion == "Confidence_margin":
        sorted_labels, sorted_logits = Confidence_sorting(labels, logits)
    elif SC_criterion == "Geo_margin":
        sorted_labels, sorted_logits = Margin_sorting(labels, logits, args['last_fc_weights'])
    else:
        exit(f"{SC_criterion} is currently not supported")
    
    
    aurc, eaurc, acc = AURC_helper(sorted_labels, sorted_logits)
    return aurc, eaurc, acc
    
def eval(logits, labels, SC_criterion="SR", **args):
    # aurc, eaurc = calc_aurc_eaurc(in_d_logits, correct)
    ## only for binary
    # aupr_err, fpr_in_tpr_95 = calc_fpr_aupr(in_d_logits, in_d_labels)
    aurc, eaurc, acc = AURC(logits, labels, SC_criterion, **args)
    baauc, ebaauc, ba = AUBAC(logits, labels, SC_criterion, **args)
    
    print(f"============ {SC_criterion} ============")
    print("Acc - %.04f" % acc)
    print("AURC - %.04f" % aurc)
    print("EAURC - %.04f" % eaurc)
    print("BAcc - %.04f" % ba)
    print("AUBAC - %.04f" % baauc)
    print("EAUBAC - %.04f" % ebaauc)
    print("\n")

def main(read_root_dir):
    # ===  Load In-D collected data ===
    in_d_logits, in_d_labels, fc_weights, fc_bias = read_data(read_root_dir, split="test_set", load_classifier_weight=True)
    # in_d_logits = in_d_logits[:, np.newaxis]
    print("Check In-D shapes: ", in_d_logits.shape, in_d_labels.shape)

    print("Check In-D shapes: ", in_d_logits.shape, in_d_labels.shape)
    acc = np.mean(np.argmax(in_d_logits, axis=1) == in_d_labels)
    
    correct = np.argmax(in_d_logits, axis=1) == in_d_labels
    eval(in_d_logits, in_d_labels, SC_criterion="SR")
    eval(in_d_logits, in_d_labels, SC_criterion="Confidence_margin")
    eval(in_d_logits, in_d_labels, SC_criterion="Geo_margin", last_fc_weights=fc_weights)
    

    
    return 
    # print("AUPR_err - %.04f" % aupr_err)
    # print("FPR_IN_TPR@95 - %.04f" % fpr_in_tpr_95)
    # break
    
    save_root_dir = read_root_dir
    
    save_rc_curve_root = os.path.join(save_root_dir, "rc_curves")
    os.makedirs(save_rc_curve_root, exist_ok=True)
    # # === Check RC and acc-coverage curve ===
    acc_scores, acc_list, acc_method_list = calculate_score_acc(in_d_logits, in_d_labels, binary_cls=True)
    acc_fig_name = os.path.join(save_rc_curve_root, "acc-coverage-curve.png")
    acc_coverage_dict, acc_dict = plot_rc_curve(
        acc_scores, acc_list, acc_fig_name, acc_method_list, PLOT_SYMBOL_DICT, curve_name="acc-coverage"
    )

    # === Check RC and acc-coverage curve ===
    risk_scores, risk_list, risk_method_list = calculate_score_residual(in_d_logits, in_d_labels, binary_cls=True)
    risk_fig_name = os.path.join(save_rc_curve_root, "risk-coverage-curve.png")
    risk_coverage_dict, risk_dict = plot_rc_curve(
        risk_scores, risk_list, risk_fig_name, risk_method_list, PLOT_SYMBOL_DICT, curve_name="risk-coverage"
    )
    

    # === Plot sample rejection dynamics ===
    scores_dict, method_names, labels_list = calculate_score_sample_cls(in_d_logits, in_d_labels, binary_cls=True)
    fig_path = os.path.join(save_root_dir, "class-wise-sample-rejection")
    os.makedirs(fig_path, exist_ok=True)
    plot_sample_percentage_coverage_curve(scores_dict, method_names, labels_list, fig_path)

    # === Plot recall-coverage curve
    fig_path = os.path.join(save_root_dir, "recall-coverage")
    os.makedirs(fig_path, exist_ok=True)
    plot_recall_coverage_curve(in_d_logits, in_d_labels, fig_path, binary_cls=True)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", dest="data", type=str,
        default="ham",
    )
    
    parser.add_argument(
        "--loss", dest="loss", type=str,
        default="ce",
    )
    
    # parser.add_argument(
    #     "--imbalance", action="store_true"

    # )
    
    args = parser.parse_args()
    
    read_root_dir = f"../../raw_data_collection/{args.data}/{args.loss}/"
    main(read_root_dir)
    