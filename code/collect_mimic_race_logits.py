import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import itertools
from tqdm import tqdm

# === add abs path for import convenience
import sys, os, argparse, time
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
from utils.utils import set_seed

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=100, num_layers=10, output_dim=2):
        super().__init__()
        bias = False
        layers = [nn.Linear(input_dim, hidden_dim, bias=bias)]
        for _ in range(num_layers-2):
            layers.append(nn.BatchNorm1d(num_features = hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
        if output_dim == 2:
            layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))
        else:
            layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))

        # layers = [nn.Linear(input_dim, 1, bias=bias)]
        # layers.append(nn.Sigmoid())


        self.net = nn.Sequential(*layers)
    def forward(self, X):
        return self.net(X)
    
    
    
def collect_logits(model, data_loader, save_res_root, device):
    model.eval()
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

    
# pretrained_model_name = 'best_model_ethnicity_exclude.pth'
# testing(MIMIC_ethnicity_exclude, pretrained_model_name)

# pretrained_model_name = 'best_model_ethnicity.pth'
# testing(MIMIC_ethnicity, pretrained_model_name)

# pretrained_model_name = 'best_model_raw_exclude.pth'
# testing(MIMIC_originl_exclude, pretrained_model_name)

# pretrained_model_name = 'best_model_raw.pth'
# testing(MIMIC_originl, pretrained_model_name)

def get_mimic_race_loader():
    MIMIC_ethnicity_exclude = pd.read_csv('/home/jusun/shared/For_HY/datasets/MIMIC/MIMIC_ethnicity_exclude.csv') # Other as the test data and remove the ethicity as the feature
    # MIMIC_ethnicity = pd.read_csv('MIMIC_ethnicity.csv') #Other as the test data
    # MIMIC_originl_exclude = pd.read_csv('MIMIC_originl_exclude.csv') # Other as the test data and remove the ethicity as the feature
    # MIMIC_originl = pd.read_csv('MIMIC_originl.csv') #Other as the test data

    test_data = MIMIC_ethnicity_exclude
    y_test = test_data["mort_icu"]
    X_test = test_data.loc[:, test_data.columns != "mort_icu"]

    testset_x = torch.from_numpy(X_test.values).float()
    testset_y = torch.from_numpy(y_test.values.ravel()).float()
    testset_y = testset_y.unsqueeze(1)
    
    testset = TensorDataset(testset_x, testset_y)
    testloader = DataLoader(testset, batch_size=128, shuffle=False)
    dss = {
        "test": testloader
    }
    stats = {
        "input_dim": X_test.shape[1]
    }
    return dss, stats

def main(args):
    name_str = args.ckpt_dir.split("/")[-2]
    corr_name = "none"

    # === Create Exp Save Root ===
    log_root = os.path.join(".", "raw_data_collection", "MIMIC_race", name_str, corr_name)
    os.makedirs(log_root, exist_ok=True)

    set_seed(args.seed) # important! For reproduction
    device = torch.device("cuda")
    
    dss, stats = get_mimic_race_loader()
    
    # Prepare Pretrained Model
    model = MLP(stats['input_dim'])
    model.load_state_dict(torch.load(args.ckpt_dir))
    model.to(device)
    # model.eval() ## move this to test function

    # === Collect Model fc weights and bias ===
    # last_layer = model[-1]
    last_layer = list(model._modules.values())[-1][-1]
    print(last_layer)
    weights = last_layer.weight.data.clone().cpu().numpy()
    save_weight_name = os.path.join(
        log_root, "last_layer_weights.npy"
    )
    np.save(save_weight_name, weights)

    
    

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
        default="/panfs/jay/groups/15/jusun/shared/For_HY/models/MIMIC/race/best_model_ethnicity_exclude.pth"
    )
    args = parser.parse_args()
    main(args)
    print("All task completed!")
    
    


