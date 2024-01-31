import torch, os
import random
import numpy as np
import platform
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, accuracy_score


def set_seed(seed):
    """Set all random seeds and settings for reproducibility (deterministic behavior)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def clear_terminal_output():
    system_os = platform.system()
    if "Windows" in system_os:
        cmd = "cls"
    elif "Linux" in system_os:
        cmd = "clear"
    os.system(cmd)


