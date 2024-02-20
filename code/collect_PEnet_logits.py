import cv2
import json
import pickle
import numpy as np
import os
import sklearn.metrics as sk_metrics
import torch
import torch.nn.functional as F
import torch.nn as nn
import time

from collections import defaultdict
from PIL import Image
from PE_utils.resnext3d101 import resnext101_layer
from PE_utils.dataset import CTDataLoader
import PE_utils as util
from tqdm import tqdm
import torch.nn.functional as F

import sys, os, argparse, time
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
dir_path = os.path.abspath(os.path.join(".", "code"))
sys.path.append(dir_path)
from utils.utils import set_seed


DATA_PATH = "/scratch.global/peng0347/CTs/"


def collect_logits(model, data_loader, save_res_root, device):
    logits_log = []
    labels_log = []

    study2labels = {}
    study2slices = defaultdict(list)
    study2probs = defaultdict(list)
    with torch.no_grad():
        for x, targets_dict in tqdm(data_loader):
            input, target = input.to(device), target.to(device)

            # compute output
            cls_logits = model(x)
            cls_probs = F.sigmoid(cls_logits)
            max_probs = cls_probs.to('cpu').numpy()

            # == Construct logged stats ==
            for study_num, slice_idx, prob in zip(
                targets_dict['study_num'], targets_dict['slice_idx'], list(max_probs)
            ):
                # Convert to standard python data types
                study_num = int(study_num)
                slice_idx = int(slice_idx)

                # Save series num for aggregation
                study2slices[study_num].append(slice_idx)
                study2probs[study_num].append(prob.item())

                series = data_loader.get_series(study_num)
                if study_num not in study2labels:
                    study2labels[study_num] = int(series.is_positive)

    logits_log = []
    labels_log = []

    for study_num in tqdm(study2slices):

        # Sort by slice index and get max probability
        slice_list, prob_list = (list(t) for t in zip(*sorted(zip(study2slices[study_num], study2probs[study_num]),
                                                              key=lambda slice_and_prob: slice_and_prob[0])))
        study2slices[study_num] = slice_list
        study2probs[study_num] = prob_list
        max_prob = max(prob_list)
        logits_log.append(max_prob)
        label = study2labels[study_num]
        labels_log.append(label)

    logits_log = np.concatenate(logits_log, axis=0)
    labels_log = np.concatenate(labels_log, axis=0)
    print("Check collected shapes -- Logits ", logits_log.shape, "  | Labels ", labels_log.shape)
    save_logits_name = os.path.join(save_res_root, "pred_logits.npy")
    np.save(save_logits_name, logits_log)
    save_labels_name = os.path.join(save_res_root, "labels.npy")
    np.save(save_labels_name, labels_log)


def main(args):
    ckpt_dir = args.ckpt_dir

    device = torch.device('cuda')
    model = resnext101_layer(truncate_level=4, num_classes=1, trunc_layer=30, pretrained=False).to(device)
    model = torch.nn.DataParallel(model) ## keep this as model are saved in this way
    ckpt_dict = torch.load(ckpt_dir, map_location=device)
    model.load_state_dict(ckpt_dict['model_state'])
    model.eval()

    data_loader = CTDataLoader(
        data_dir=DATA_PATH, batch_size=16, phase="test", is_training=False
    )

    # === Create Exp Save Root ===
    name_str = args.ckpt_dir.split("/")[-2]
    log_root = os.path.join(".", "raw_data_collection", "PE_net", "%s" % name_str)
    os.makedirs(log_root, exist_ok=True)

    save_dir = os.path.join(log_root, "test_set")
    os.makedirs(save_dir, exist_ok=True)
    collect_logits(model, data_loader, save_dir, device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", dest="seed", type=int,
        default=0,
        help="Random seed."
    )
    parser.add_argument(
        "--ckpt_dir", dest="ckpt_dir", type=str,
        default="/panfs/jay/groups/15/jusun/shared/For_HY/SC_eval/models/PE/best.pth.tar"
    )
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
    print("All task completed!")