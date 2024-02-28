import torch
from torchvision import models
import collections
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision import transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
from collections import Counter
from PIL import Image
import torch.nn.functional as F

from albumentations import ( Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip, 
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout, IAAAdditiveGaussianNoise, Transpose, ToGray )
from albumentations.augmentations.transforms import CLAHE
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2
from tqdm import tqdm

from utils.corruptions import corrupt_image

# === add abs path for import convenience
import sys, os, argparse, time
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
from utils.utils import set_seed


HAM_CSV_DIR = "/panfs/jay/groups/15/jusun/shared/HAM/HAM10000.csv"


def stratfy_sampling(labelList, ratio, return_mask=False):
    """
    stratify sampling
    input: 
        labelList: list of labels to be sampled
        ratio: ratio of the train and test split (ratio = train/test)
        return_mask: bool value indicating the return type as a indices (False) or mask (True)
    """
    class_dict = {k:np.where(labelList==k)[0] for k in set(labelList)}
    test_idx = []
    for v in class_dict.values():
        np.random.shuffle(v)
        test_num = int(len(v) * ratio)
        test_idx.extend(v[:test_num])
    if return_mask:
        test_mask = np.zeros(len(labelList)).astype(int)
        test_mask[test_idx] = 1
        return 1-test_mask, test_mask
    else:
        return list(set(range(len(labelList)))- set(test_idx)), test_idx

class SquarePad(nn.Module):
    """squre input image and pad with 0s
    """
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')

class HAM_224_dataset(Dataset):
    def __init__(self, pathList: list, labelList: list, mode: str, corruption_type: str="none", severity: int=1) -> None:
        """init function

        Args:
            pathList (list): list of path to the images
            labelList (list): list of labels
            mode (str): dataset type 'train' or 'val'
        """
        super(HAM_224_dataset, self).__init__()
        assert mode in {"train", "val"}
        
        self.pathList = pathList
        self.labelList = labelList
        self.mode = mode
        self.corruption_type = corruption_type
        self.severity = severity
        

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        self.transform = {
            "train": A.Compose([
                Resize(256, 256, always_apply=True),
                RandomCrop(width=224, height=224),
                Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]),
            "val": A.Compose([
                Resize(224, 224),
                Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        }
    
    def __len__(self):
        return len(self.labelList)
    
    def __getitem__(self, idx):
        img = Image.open(self.pathList[idx]).convert("RGB")
        img = corrupt_image(img, self.corruption_type, self.severity)
        img = self.transform[self.mode](image=np.array(img))
        return img['image'], self.labelList[idx]

lesion_to_num = {'nv': 0,
        'mel': 1,
        'bkl': 2,
        'bcc': 3,
        'akiec': 4,
        'vasc': 5,
        'df': 6}


def get_ham_loaders(bs=128):
    df = pd.read_csv(HAM_CSV_DIR)
    df.dx = df.dx.map(lambda x: lesion_to_num[x])
    weights = list(dict(sorted(Counter(df.dx).items(), key=lambda x: x[0])).values())
    idx_train, idx_test = stratfy_sampling(df.dx, ratio=0.2)
    df_test = df.iloc[idx_test].reset_index(drop=True)
    df_tmp = df.iloc[idx_train].reset_index(drop=True)

    idx_train, idx_val = stratfy_sampling(df_tmp.dx, ratio=0.2)
    df_val = df_tmp.iloc[idx_val].reset_index(drop=True)
    df_train = df_tmp.iloc[idx_train].reset_index(drop=True)
    print(df.shape, df_train.shape, df_test.shape, df_val.shape)


    train_ds = HAM_224_dataset(df_train.image_id, df_train.dx, mode='train')
    val_ds = HAM_224_dataset(df_val.image_id, df_val.dx, mode='val')
    test_ds = HAM_224_dataset(df_test.image_id, df_test.dx, mode='val')
    dss = {'train': train_ds, 'val': val_ds, 'test': test_ds}

    trainloader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=False)
    valloader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=False)
    testloader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=False)
    
    dls = {'train': trainloader, 'val': valloader, 'test': testloader} 

    ## get dataset stats
    stats = {}
    for stage, tmp_df in zip(["full", "train", "val", "test"], [df, df_train, df_val, df_test]):
        stats.update({
            stage: {
                "size": tmp_df.shape[0],
                "label distribution": dict(Counter(tmp_df.dx).most_common())
            }
        })
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
    name_str = args.ckpt_dir.split("/")[-2]
    corr_name = "clean" if args.corrupt == "none" else args.corrupt

    # === Create Exp Save Root ===
    log_root = os.path.join(".", "raw_data_collection", "HAM", name_str, corr_name)
    os.makedirs(log_root, exist_ok=True)

    set_seed(args.seed) # important! For reproduction
    device = torch.device("cuda")
    
    # Prepare Pretrained Model
    num_classes = 7
    model = models.resnet50()
    backbone=nn.Sequential(*list(model.children())[:-1], nn.Flatten())
    model = torch.nn.Sequential(
        collections.OrderedDict([
                ("backbone", backbone),
                ("fc", nn.Linear(model.fc.in_features, num_classes))
            ]
        )
    )
    model.load_state_dict(torch.load(args.ckpt_dir))
    model.to(device)
    model.eval()

    # === Collect Model fc weights and bias ===
    last_layer = model[-1]
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

    dss, stats = get_ham_loaders()
    
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
        "--corrupt", dest="corrupt", type=str,
        default="none",
        help="Corruption type."
    )
    parser.add_argument(
        "--severity", dest="severity", type=int,
        default=1,
        help="Corruption severity."
    )
    parser.add_argument(
        "--ckpt_dir", dest="ckpt_dir", type=str,
        default="/panfs/jay/groups/15/jusun/shared/For_HY/SC_eval/models/HAM/ce/final.pt"
    )
    args = parser.parse_args()
    main(args)
    print("All task completed!")
        
