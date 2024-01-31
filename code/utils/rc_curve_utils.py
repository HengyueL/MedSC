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

#  reverse of residual, to be consistent with risk
def calculate_acc(pred, label):
    pred_tensor = torch.from_numpy(pred)
    label_tensor = torch.from_numpy(label)
    predict_correct_bool = pred_tensor == label_tensor
    correct = torch.where(predict_correct_bool, 1, 0)
    return correct.cpu().numpy()


# Get score - pred_correct pairs
def calculate_score_acc(
        logits, labels, num_classes=10,
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