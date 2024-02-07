import argparse
import os
import pandas as pd
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader, Dataset
import os
import torchvision
import cv2
import torch
from torchvision import models
import torch.nn as nn
# === add abs path for import convenience
import sys, os, argparse, time
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
from utils.utils import set_seed


def main(args):
    # === Create Exp Save Root ===
    log_root = os.path.join(".", "raw_data_collection", "NIH-res50")
    os.makedirs(log_root, exist_ok=True)

    set_seed(args.seed) # important! For reproduction
    device = torch.device("cuda")

    # Prepare Model 
    num_classes= 7   
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 20)
    weights = torch.load(args.ckpt_dir)['weights']
    model.load_state_dict(weights)
    model.to(device)
    model.eval()


    # === Collect Model fc weights and bias ===
    last_layer = model.fc
    weights = last_layer.weight.data.clone().cpu().numpy()
    bias = last_layer.bias.data.clone().cpu().numpy()
    save_weight_name = os.path.join(
        log_root, "last_layer_weights.npy"
    )
    save_bias_name = os.path.join(
        log_root, "last_layer_bias.npy"
    )
    np.save(save_weight_name, weights)
    np.save(save_bias_name, bias)

    dss, stats = get_NIH_TL_dataloader()

    # === Collect Training Logits === 
    train_loader = dss["train"]
    save_dir = os.path.join(log_root, "train_set")
    os.makedirs(save_dir, exist_ok=True)
    collect_logits(model, train_loader, save_dir, device)

    # === Collect Training Logits === 
    val_loader = dss["val"]
    save_dir = os.path.join(log_root, "val_set")
    os.makedirs(save_dir, exist_ok=True)
    collect_logits(model, val_loader, save_dir, device)

    # === Collect Training Logits === 
    test_loader = dss["test"]
    save_dir = os.path.join(log_root, "test_set")
    os.makedirs(save_dir, exist_ok=True)
    collect_logits(model, test_loader, save_dir, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", dest="seed", type=int,
        default=0,
        help="Random seed."
    )
    parser.add_argument(
        "--ckpt_dir", dest="ckpt_dir", type=str,
        default="/panfs/jay/groups/15/jusun/shared/For_HY/SC_eval/models/NIH/best.pt"
    )
    args = parser.parse_args()
    main(args)
    print("All task completed!")