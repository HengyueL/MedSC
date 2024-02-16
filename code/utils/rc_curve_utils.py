from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, accuracy_score
import numpy as np
import torch
import torch.nn as nn
import os
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as mcolors
import seaborn as sns
sns.set()
COLORS = list(mcolors.TABLEAU_COLORS)
N_COLORS = len(COLORS)


def compute_recalls(pred_logits, labels):
    # get onehot binary predictions
    num_classes = pred_logits.shape[1]
    num_data = pred_logits.shape[0]

    binary_pred = np.zeros_like(pred_logits)
    binary_pred[np.arange(num_data), np.argmax(pred_logits, axis=1)] = 1
    
    recall_dict = {}
    for i in range(num_classes):
        recall_dict[i] = recall_score(y_true=(labels==i), y_pred=binary_pred[:, i], zero_division=0)

    # Compute Balanced-accuracy, i.e., avg recall
    recalls = []
    for key in recall_dict.keys():
        recalls.append(recall_dict[key])
    
    recall_dict["avg_recall"] = np.mean(recalls)
    return recall_dict


# === ACC - coverage functions ===
def calculate_acc(pred, label):
    pred_tensor = torch.from_numpy(pred)
    label_tensor = torch.from_numpy(label)
    predict_correct_bool = pred_tensor == label_tensor
    correct = torch.where(predict_correct_bool, 1, 0)
    return correct.cpu().numpy()


# Get score - pred_correct pairs for RC curve
def calculate_score_acc(
        logits, labels,
        weights=None, bias=None  # reserved options in case we need geo margin
    ):
    method_name_list = []
    scores_dict = {}
    residuals_dict = {}

    # === Scores used in previous version ===
    logits_tensor = torch.from_numpy(logits).to(dtype=torch.float)
    max_logit_pred = np.argmax(logits, axis=1)


    # === MaxSR 
    sr = torch.softmax(logits_tensor, dim=1)
    max_sr_scores = torch.amax(sr, dim=1).numpy()
   
    max_sr_pred = max_logit_pred
    max_sr_acc = calculate_acc(max_sr_pred, labels)
    method_name = "max_sr"
    method_name_list.append(method_name)
    scores_dict[method_name] = max_sr_scores
    residuals_dict[method_name] = max_sr_acc

    # === OURS ====
    # raw margin
    values, indices = torch.topk(logits_tensor, 2, axis=1)
    raw_margin_scores = (values[:, 0] - values[:, 1]).cpu().numpy()
    raw_margin_pred = max_logit_pred
    raw_margin_acc = calculate_acc(raw_margin_pred, labels)
    method_name = "conf_margin"
    method_name_list.append(method_name)
    scores_dict[method_name] = raw_margin_scores
    residuals_dict[method_name] = raw_margin_acc

    return scores_dict, residuals_dict, method_name_list


def acc_coverage_curve(acc_list, confidence):
    """
        This function turns the model's 1) confidence prediction --- (n, ) ndarray; 2) acc --- (n, ) ndarray
        into coverage - risk (for RC curve plots).

        acc_list: array of 0's and 1's, e.g., [0, 1, 0, 1, ... ], where 1 ==> prediction correct and 0 ==> prediction wrong
        confidence: array of the selection cofidence scores [s_1, s_2, s_3, s_4, ...]

        return:
            coverage -- (n, ), array of values from 0 to 1 for RC curve plot
            risk -- (n, ) array of selection risk corr. to the coverage point.
    """

    curve = []
    m = len(acc_list)
    idx_sorted = np.argsort(confidence)
    temp1 = acc_list[idx_sorted]
    cov = len(temp1)
    acc_total = sum(temp1)
    curve.append((cov/ m, acc_total / len(temp1)))
    for i in range(0, len(idx_sorted)-1):
        cov = cov-1
        acc_total = acc_total - acc_list[idx_sorted[i]]
        curve.append((cov / m, acc_total /(m-i-1)))
    
    # AUC = sum([a[1] for a in curve])/len(curve)
    # err = np.mean(residuals)
    # kappa_star_aurc = err + (1 - err) * (np.log(1 - err))
    # EAURC = AUC-kappa_star_aurc
    # return curve, AUC, EAURC

    curve = np.asarray(curve)
    coverage, sc_acc = curve[:, 0], curve[:, 1]
    return coverage, sc_acc


# === RC curve functions ====
def calculate_residual(pred, label):
    pred_tensor = torch.from_numpy(pred)
    label_tensor = torch.from_numpy(label)
    predict_correct_bool = pred_tensor == label_tensor
    residual_tensor = torch.where(predict_correct_bool, 0, 1)
    return residual_tensor.cpu().numpy()


def calculate_score_residual(
        logits, labels, weight_norm=None
    ):
    scores_dict = {}
    residuals_dict = {}
    method_name_list = []

    # === Scores used in previous version ===
    logits_tensor = torch.from_numpy(logits).to(dtype=torch.float)
    max_logit_pred = np.argmax(logits, axis=1)

    # === MaxSR ===
    sr = torch.softmax(logits_tensor, dim=1)
    max_sr_scores = torch.amax(sr, dim=1).numpy()
   
    max_sr_pred = max_logit_pred
    max_sr_residuals = calculate_residual(max_sr_pred, labels)
    method_name = "max_sr"
    method_name_list.append(method_name)
    scores_dict[method_name] = max_sr_scores
    residuals_dict[method_name] = max_sr_residuals
        
    # raw margin
    values, indices = torch.topk(logits_tensor, 2, axis=1)
    raw_margin_scores = (values[:, 0] - values[:, 1]).cpu().numpy()
    raw_margin_pred = max_logit_pred
    raw_margin_residuals = calculate_residual(raw_margin_pred, labels)
    method_name = "conf_margin"
    method_name_list.append(method_name)
    scores_dict[method_name] = raw_margin_scores
    residuals_dict[method_name] = raw_margin_residuals
    return scores_dict, residuals_dict, method_name_list


def RC_curve(residuals, confidence):
    """
        This function turns the model's 1) confidence prediction --- (n, ) ndarray; 2) residuals --- (n, ) ndarray
        into coverage - risk (for RC curve plots).

        residuals: array of 0's and 1's, e.g., [0, 1, 0, 1, ... ], where 0 ==> prediction correct and 1 ==> prediction wrong
        confidence: array of the selection cofidence scores [s_1, s_2, s_3, s_4, ...]

        return:
            coverage -- (n, ), array of values from 0 to 1 for RC curve plot
            risk -- (n, ) array of selection risk corr. to the coverage point.
    """

    curve = []
    m = len(residuals)
    idx_sorted = np.argsort(confidence)
    temp1 = residuals[idx_sorted]
    cov = len(temp1)
    acc = sum(temp1)
    curve.append((cov/ m, acc / len(temp1)))
    for i in range(0, len(idx_sorted)-1):
        cov = cov-1
        acc = acc-residuals[idx_sorted[i]]
        curve.append((cov / m, acc /(m-i-1)))
    
    # AUC = sum([a[1] for a in curve])/len(curve)
    # err = np.mean(residuals)
    # kappa_star_aurc = err + (1 - err) * (np.log(1 - err))
    # EAURC = AUC-kappa_star_aurc
    # return curve, AUC, EAURC

    curve = np.asarray(curve)
    coverage, risk = curve[:, 0], curve[:, 1]
    return coverage, risk


def select_RC_curve_points(coverage_x, risk_y, n_plot_points=40,  min_n_samples=-10):

    plot_interval = len(coverage_x) // n_plot_points
    coverage_plot, risk_plot = coverage_x[0::plot_interval].tolist(), risk_y[0::plot_interval].tolist()
    coverage_plot.append(coverage_x[min_n_samples])
    risk_plot.append(risk_y[min_n_samples])
    return coverage_plot, risk_plot


def plot_rc_curve(total_scores_dict, risk_acc_dict, fig_name, method_name_list, plot_symbol_dict, curve_name="risk-coverage"):
    coverage_dict, y_dict = {}, {}

    if curve_name == "risk-coverage":
        for method_name in method_name_list:
            x, y = RC_curve(
                risk_acc_dict[method_name], total_scores_dict[method_name]
            )
            coverage_dict[method_name] = x
            y_dict[method_name] = y
    elif curve_name == "acc-coverage":
        for method_name in method_name_list:
            x, y = acc_coverage_curve(
                risk_acc_dict[method_name], total_scores_dict[method_name]
            )
            coverage_dict[method_name] = x
            y_dict[method_name] = y

    else:
        raise RuntimeError("Somthing Happened")

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
    for method_name in method_name_list:
        coverage_plot, y_plot = coverage_dict[method_name], y_dict[method_name]
        # x_plot, y_plot = select_RC_curve_points(coverage_plot, y_plot, plot_n_points, min_num_samples)
        x_plot, y_plot = coverage_plot, y_plot
        y_max, y_min = max(y_plot[0], y_max), min(np.amin(y_plot), y_min)
        # y_max, y_min = max(np.amax(y_plot), y_max), min(np.amin(y_plot), y_min)
        plot_settings = plot_symbol_dict[method_name]
        ax.plot(
            x_plot, y_plot,
            label=plot_settings[2], lw=line_width, alpha=alpha,
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
    return coverage_dict, y_dict


# === calculate score and cls index === To plot which samples are rejected first ===
def calculate_score_sample_cls(
        logits, labels,
        weights=None, bias=None  # reserved options in case we need geo margin
    ):
    method_name_list = []
    scores_dict = {}

    # === Scores used in previous version ===
    logits_tensor = torch.from_numpy(logits).to(dtype=torch.float)
    max_logit_pred = np.argmax(logits, axis=1)


    # === MaxSR 
    sr = torch.softmax(logits_tensor, dim=1)
    max_sr_scores = torch.amax(sr, dim=1).numpy()
    method_name = "max_sr"
    method_name_list.append(method_name)
    scores_dict[method_name] = max_sr_scores


    # === OURS ====
    # raw margin
    values, _ = torch.topk(logits_tensor, 2, axis=1)
    raw_margin_scores = (values[:, 0] - values[:, 1]).cpu().numpy()
    method_name = "conf_margin"
    method_name_list.append(method_name)
    scores_dict[method_name] = raw_margin_scores

    return scores_dict, method_name_list, labels.tolist()

 
def sample_percentage_coverage_curve(scores_list, n_samples_dict, labels_list):
    """
        Compute the (relative) samples remained when coverage decreases.
    """
    # return result
    coverage_dict = {}
    sample_percentage_dict = {}
    sample_abs_number_dict = {}

    # counter
    samples_remained = {}
    n_samples = len(labels_list)
    n_samples_remained = n_samples

    for key in n_samples_dict:
        samples_remained[key] = n_samples_dict[key]
        coverage_dict[key] = [1]
        sample_percentage_dict[key] = [1]
        sample_abs_number_dict[key] = [n_samples_dict[key]]
        
    idx_sorted = np.argsort(scores_list)
    for i in range(0, len(idx_sorted)-1):
        poped_idx = idx_sorted[i]
        reject_sample_label = labels_list[poped_idx]

        samples_remained[reject_sample_label] -= 1
        n_samples_remained -= 1
        
        for key in n_samples_dict:
            coverage = n_samples_remained / n_samples
            ratio = samples_remained[key] / n_samples_dict[key]

            coverage_dict[key].append(coverage)
            sample_percentage_dict[key].append(ratio)
            sample_abs_number_dict[key].append(samples_remained[key])
    return coverage_dict, sample_percentage_dict, sample_abs_number_dict
    

def plot_sample_percentage_coverage_curve(scores_dict, method_name_list, labels_list, fig_path):
    n_samples_dict = dict(Counter(labels_list))

    for method in method_name_list:
        scores_list = scores_dict[method]
        coverage_dict, sample_percentage_dict, sample_abs_number_dict = sample_percentage_coverage_curve(
            scores_list, n_samples_dict, labels_list
        )  # Per cls coverage-percentage curve

        # === Plot the curve ===
        save_path = os.path.join(fig_path, "sample_rejections_%s.png" % method)
        line_width = 2
        markersize = 1
        alpha = 0.5
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        font_size = 19
        tick_size = 20
        for curve_idx, key in enumerate(coverage_dict.keys()):
            x_plot, y_plot = coverage_dict[key], sample_percentage_dict[key]
            l1 = ax[0].plot(
                x_plot, y_plot,
                label="Cls - %s" % key, lw=line_width, alpha=alpha,
                color=COLORS[curve_idx % N_COLORS], marker=None, markersize=markersize
            )

            x_plot, y_plot = coverage_dict[key], sample_abs_number_dict[key]
            _ = ax[1].plot(
                x_plot, y_plot,
                label="Cls - %s" % key, lw=line_width, alpha=alpha,
                color=COLORS[curve_idx % N_COLORS], marker=None, markersize=markersize
            )
        
        ax[0].legend(
            loc='lower left', bbox_to_anchor=(-0.25, 1, 1.25, 0.2), mode="expand", 
            borderaxespad=0,
            ncol=3, fancybox=True, shadow=False, fontsize=font_size, framealpha=0.3
        )
        ax[0].tick_params(axis='x', which='major', colors='black', labelsize=tick_size)
        ax[0].tick_params(axis='y', which='major', colors='black', labelsize=tick_size)
        ax[0].set_ylabel(r"Samples remained (%)", fontsize=font_size)
        ax[0].set_xlabel(r"Coverage", fontsize=font_size)

        ax[1].legend(
            loc='lower left', bbox_to_anchor=(-0.25, 1, 1.25, 0.2), mode="expand", 
            borderaxespad=0,
            ncol=3, fancybox=True, shadow=False, fontsize=font_size, framealpha=0.3
        )
        ax[1].tick_params(axis='x', which='major', colors='black', labelsize=tick_size)
        ax[1].tick_params(axis='y', which='major', colors='black', labelsize=tick_size)
        ax[1].set_ylabel(r"Samples remained (abs number)", fontsize=font_size)
        ax[1].set_xlabel(r"Coverage", fontsize=font_size)

        fig.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)


def recall_coverage_curve(scores_list, logits, labels):
    idx_sorted = np.argsort(scores_list)
    n_samples = len(idx_sorted)
    # === For result saving ===
    recall_dict = {}
    coverage_list = [1]

    init_recalls = compute_recalls(logits, labels)
    for key in init_recalls.keys():
        recall_dict[key] = [init_recalls[key]]

    for i in range(0, len(idx_sorted)-1):
        indices = idx_sorted[i:]
        logits_remained = logits[indices, :]
        labels_remained = labels[indices]
        recalls = compute_recalls(logits_remained, labels_remained)
        # log result
        coverage_list.append(len(labels_remained)/n_samples)
        for key in init_recalls.keys():
            recall_dict[key].append(recalls[key])

    return coverage_list, recall_dict


def plot_recall_coverage_curve(logits, labels, fig_path):
    scores_dict, _, method_names = calculate_score_acc(logits, labels)

    for method in method_names:
        scores_list = scores_dict[method]

        coverage_list, recall_dict = recall_coverage_curve(
            scores_list, logits, labels
        )

        # === Plot Curve ===
        save_path = os.path.join(fig_path, "recalls_%s.png" % method)
        line_width = 2
        markersize = 1
        alpha = 0.5
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 12))
        font_size = 19
        tick_size = 20

        for curve_idx, key in enumerate(recall_dict.keys()):
            x_plot, y_plot = coverage_list, recall_dict[key]
            
            if type(key) == int:
                legend_str = r"Cls - %s" % key 
                l1 = ax[0].plot(
                    x_plot, y_plot,
                    label=legend_str, lw=line_width, alpha=alpha,
                    color=COLORS[curve_idx % N_COLORS], marker=None, markersize=markersize
                )
            else:
                legend_str = r"Balance Acc."
                l1 = ax[1].plot(
                    x_plot, y_plot,
                    label=legend_str, lw=line_width, alpha=alpha,
                    color=COLORS[curve_idx % N_COLORS], marker=None, markersize=markersize
                )

        ax[0].legend(
            loc='lower left', bbox_to_anchor=(-0.25, 1, 1.25, 0.2), mode="expand", 
            borderaxespad=0,
            ncol=3, fancybox=True, shadow=False, fontsize=font_size, framealpha=0.3
        )
        ax[0].tick_params(axis='x', which='major', colors='black', labelsize=tick_size)
        ax[0].tick_params(axis='y', which='major', colors='black', labelsize=tick_size)
        ax[0].set_ylabel(r"Recalls", fontsize=font_size)
        ax[0].set_xlabel(r"Coverage", fontsize=font_size)

        ax[1].legend(
            loc='lower left', bbox_to_anchor=(-0.25, 1, 1.25, 0.2), mode="expand", 
            borderaxespad=0,
            ncol=3, fancybox=True, shadow=False, fontsize=font_size, framealpha=0.3
        )
        ax[1].tick_params(axis='x', which='major', colors='black', labelsize=tick_size)
        ax[1].tick_params(axis='y', which='major', colors='black', labelsize=tick_size)
        ax[1].set_ylabel(r"Balance Acc.", fontsize=font_size)
        ax[1].set_xlabel(r"Coverage", fontsize=font_size)

        fig.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)