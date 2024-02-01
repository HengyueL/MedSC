from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, accuracy_score
import numpy as np
import torch
import torch.nn as nn


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
    scores_dict[method_name] = max_sr_scores
    residuals_dict[method_name] = max_sr_acc

    # === OURS ====
    # raw margin
    values, indices = torch.topk(logits_tensor, 2, axis=1)
    raw_margin_scores = (values[:, 0] - values[:, 1]).cpu().numpy()
    raw_margin_pred = max_logit_pred
    raw_margin_acc = calculate_acc(raw_margin_pred, labels)
    method_name = "conf_margin"
    scores_dict[method_name] = raw_margin_scores
    residuals_dict[method_name] = raw_margin_acc

    return scores_dict, residuals_dict


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
        acc = acc_total - acc_list[idx_sorted[i]]
        curve.append((cov / m, acc /(m-i)))
    
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

    # === Scores used in previous version ===
    logits_tensor = torch.from_numpy(logits).to(dtype=torch.float)
    max_logit_pred = np.argmax(logits, axis=1)


    # === MaxSR ===
    sr = torch.softmax(logits_tensor, dim=1)
    max_sr_scores = torch.amax(sr, dim=1).numpy()
   
    max_sr_pred = max_logit_pred
    max_sr_residuals = calculate_residual(max_sr_pred, labels)
    method_name = "max_sr"
    scores_dict[method_name] = max_sr_scores
    residuals_dict[method_name] = max_sr_residuals
        
    # raw margin
    values, indices = torch.topk(logits_tensor, 2, axis=1)
    raw_margin_scores = (values[:, 0] - values[:, 1]).cpu().numpy()
    raw_margin_pred = max_logit_pred
    raw_margin_residuals = calculate_residual(raw_margin_pred, labels)
    method_name = "raw_margin"
    scores_dict[method_name] = raw_margin_scores
    residuals_dict[method_name] = raw_margin_residuals

    return scores_dict, residuals_dict