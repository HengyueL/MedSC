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
from tqdm import tqdm
# === add abs path for import convenience
import sys, os, argparse, time
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
from utils.utils import set_seed


def parse_df(df, one_hot=False):
    path = "/scratch.global/peng0347/nih-crx-lt/images/images/" + df.id
    
    labels = df.iloc[:, 1:21].to_numpy()
    
    if not one_hot:
        labels = labels @ np.asarray(range(0, 20))
    return path, labels


class NIH_224_dataset_fromdf(Dataset):
    def __init__(self, mode: str, split: str, size: int) -> None:
        """init function

        Args:
            pathList (list): list of path to the images
            labelList (list): list of labels
            mode (str): dataset type 'train' or 'val'
        """
        super(NIH_224_dataset_fromdf, self).__init__()
        assert mode in {"train", "val"}
        
        self.CLASSES = [
            'No Finding', 'Infiltration', 'Atelectasis', 'Effusion', 'Nodule',
            'Mass', 'Pneumothorax', 'Consolidation', 'Pleural_Thickening',
            'Cardiomegaly', 'Fibrosis', 'Edema', 'Tortuous Aorta', 'Emphysema',
            'Pneumonia', 'Calcification of the Aorta', 'Pneumoperitoneum', 'Hernia',
            'Subcutaneous Emphysema', 'Pneumomediastinum'
        ]
        self.mode = mode
        self.label_df = pd.read_csv(os.path.join("/scratch.global/peng0347/nih-crx-lt/LongTailCXR", f'nih-cxr-lt_single-label_{split}.csv'))

        self.img_paths = self.label_df['id'].apply(lambda x: os.path.join("/scratch.global/peng0347/nih-crx-lt/images/images", x)).values.tolist()
        self.labels = self.label_df[self.CLASSES].idxmax(axis=1).apply(lambda x: self.CLASSES.index(x)).values

        self.cls_num_list = self.label_df[self.CLASSES].sum(0).values.tolist()
        self.size = size

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        self.transform = {
            "train": torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ]),
            "val": torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])
        }
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # img = Image.open(self.pathList[idx]).convert("RGB")
        x = cv2.imread(self.img_paths[idx])
        x = cv2.resize(x, (self.size, self.size), interpolation=cv2.INTER_AREA)
        x = self.transform[self.mode](x)
        y = np.array(self.labels[idx])

        return x.float(), torch.from_numpy(y).long()


def get_NIH_TL_dataloader(bs=256, size=224):
    train_df = pd.read_csv("/scratch.global/peng0347/nih-crx-lt/LongTailCXR/nih-cxr-lt_single-label_train.csv")
    val_df = pd.read_csv("/scratch.global/peng0347/nih-crx-lt/LongTailCXR/nih-cxr-lt_single-label_balanced-val.csv")
    test_df = pd.read_csv("/scratch.global/peng0347/nih-crx-lt/LongTailCXR/nih-cxr-lt_single-label_test.csv")
    btest_df = pd.read_csv("/scratch.global/peng0347/nih-crx-lt/LongTailCXR/nih-cxr-lt_single-label_balanced-test.csv")
    
    train_path, train_labels = parse_df(train_df)
    val_path, val_labels = parse_df(val_df)
    test_path, test_labels = parse_df(test_df)
    btest_path, btest_labels = parse_df(btest_df)
    
    # train_ds = NIH_224_dataset(train_path, train_labels, mode='train')
    # val_ds = NIH_224_dataset(val_path, val_labels, mode='val')
    # test_ds = NIH_224_dataset(test_path, test_labels, mode='val')
    
    train_ds = NIH_224_dataset_fromdf(mode='train', split="train", size=size)
    val_ds = NIH_224_dataset_fromdf(mode='val', split="balanced-val", size=size)
    test_ds = NIH_224_dataset_fromdf(mode='val', split='test', size=size)
    btest_ds = NIH_224_dataset_fromdf(mode='val', split='balanced-test', size=size)
    # dss = {'train': train_ds, 'val': val_ds, 'test': test_ds}

    trainloader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    valloader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    testloader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    btestloader = DataLoader(btest_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    
    dls = {'train': trainloader, 'val': valloader, 'test': testloader, 'btest': btestloader} 

    ## get dataset stats
    stats = {}
    for stage, labels in zip(["train", "val", "test", "btest"], [train_labels, val_labels, test_labels, btest_labels]):
        stats.update({
            stage: {
                "size": labels.shape[0],
                "label distribution": dict(Counter(labels).most_common())
            }
        })
    stats['cls_num_list'] = Counter(train_labels)
    return dls, stats


def collect_logits(model, data_loader, save_res_root, device):
    logits_log = []
    labels_log = []
    with torch.no_grad():
        for input, target in tqdm(data_loader):
            input, target = input.to(device), target.to(device)

            # compute output
            logit_output = model(input)
            # == Construct logged stats ==
            logits = logit_output.cpu().numpy()
            labels = target.cpu().numpy()

            logits_log.append(logits)
            labels_log.append(labels)
    logits_log = np.concatenate(logits_log, axis=0)
    labels_log = np.concatenate(labels_log, axis=0)
    print("Check collected shapes -- Logits ", logits_log.shape, "  | Labels ", labels_log.shape)
    save_logits_name = os.path.join(save_res_root, "pred_logits.npy")
    np.save(save_logits_name, logits_log)
    save_labels_name = os.path.join(save_res_root, "labels.npy")
    np.save(save_labels_name, labels_log)



def main(args):
    ckpt_name = "NIH-" + args.ckpt_file.split("_")[0]

    # === Create Exp Save Root ===
    log_root = os.path.join(".", "raw_data_collection", ckpt_name)
    os.makedirs(log_root, exist_ok=True)

    set_seed(args.seed) # important! For reproduction
    device = torch.device("cuda")

    # Prepare Model 
    num_classes= 7   
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 20)
    file_path = os.path.join(args.ckpt_dir, args.ckpt_file)
    weights = torch.load(file_path)['weights']
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
        "--ckpt_root_dir", dest="ckpt_dir", type=str,
        default="/panfs/jay/groups/15/jusun/shared/For_HY/SC_eval/models/NIH/"
    )
    parser.add_argument(
        "--ckpt_file", dest="ckpt_file", type=str, default="decoupling-cRT_nih.pt"
    )
    args = parser.parse_args()
    main(args)
    print("All task completed!")