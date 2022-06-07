import os
import time
import argparse
import numpy as np
import torch
from utils.utils import *
from utils.data_loader import *
from utils.evaluator import *
from config import Config
from torchvision.utils import save_image
from net.qanet import QANet
import warnings
warnings.filterwarnings("ignore")

np.random.seed(216)
torch.manual_seed(216)
torch.cuda.manual_seed_all(216)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--modules', default=12) # 8, 10, 12
parser.add_argument('--resblocks', default=8) # 6, 8, 10
parser.add_argument('--split', default='RED') # RED, NIR
parser.add_argument('--sigmoid', default=True) # True, False
parser.add_argument('--arch', default='qanet') # qanet, qanet_noqm, qanet_nowm
parser.add_argument('--model_name') # name of model to test
param = parser.parse_args()

# Import config
config = Config()
config.N_modules = int(param.modules)
config.N_resblocks = int(param.resblocks)
config.qem_sigmoid = bool(param.sigmoid)

data_directory = config.path_prefix
baseline_cpsnrs = readBaselineCPSNR(os.path.join(data_directory, "norm.csv"))
train_set_directories = getImageSetDirectories(
    os.path.join(data_directory, "train"), param.split)

if param.split == 'all':
    raise RuntimeError('Invalid split: {}. Select NIR or RED'.format(param.split))
else: 
    split = param.split

beta = config.beta

val_set_directories = getImageSetDirectories(
    os.path.join(data_directory, "val"), split)

val_dataset = ImagesetDataset(
    imset_dir=val_set_directories, patch_size=128, top_k=None, beta=beta)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False, drop_last=False, collate_fn=collateFunction(), pin_memory=True)

# Create model

if param.arch == 'qanet':
    from net.qanet import QANet
elif param.arch == 'qanet_noqm':
    from net.qanet_noqm import QANet 
elif param.arch == 'qanet_nowm':
    from net.qanet_nowm import QANet 

model = QANet(config)
model.load_state_dict(torch.load(param.model_name))
model = model.cuda()

model.eval()

with torch.no_grad():
    psnr_vals = []
    scores = []
    for val_step, (lrs, qms, hrs, hr_maps, names) in enumerate(tqdm(val_loader)):
        x_lr = lrs.float().to(config.device)
        x_qm = qms.float().to(config.device)
        x_hr = hrs.float().to(config.device).unsqueeze(1)
        mask = hr_maps.float().to(config.device).unsqueeze(1)
        
        mu_sr = model(x_lr, x_qm)

        mu_sr = np.clip(mu_sr.cpu().detach().numpy()[0].astype(np.float64), 0, 1)
        x_hr = x_hr.cpu().detach().numpy()[0]
        mask = mask.cpu().detach().numpy()[0]

        psnr = shift_cPSNR(mu_sr, x_hr, mask)
        psnr_vals.append(psnr)
        scores.append(baseline_cpsnrs[names[0]] / psnr)

    print('{} cPSNR: {} Score: {}'.format(split, np.mean(psnr_vals), np.mean(scores)))