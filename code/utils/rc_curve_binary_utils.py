import torch
import numpy as np


def calculate_acc(pred, label):
    pred_tensor = torch.from_numpy(pred)
    label_tensor = torch.from_numpy(label)
    predict_correct_bool = pred_tensor == label_tensor
    correct = torch.where(predict_correct_bool, 1, 0)
    return correct.cpu().numpy()


# Get score - pred_correct pairs for RC curve
def calculate_score_acc_binary(
        logits, labels,
        weights=None, bias=None,  # reserved options in case we need geo margin
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