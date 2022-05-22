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
parser.add_argument('--save_dir', default='results/saved_models', help='Trained model directory')
param = parser.parse_args()

model_time = time.strftime("%Y%m%d_%H%M")

# Import config
config = Config()

# Import datasets
data_directory = config.path_prefix
baseline_cpsnrs = readBaselineCPSNR(os.path.join(data_directory, "norm.csv"))
val_RED_directories = getImageSetDirectories(
    os.path.join(data_directory, "val"), "RED")
val_NIR_directories = getImageSetDirectories(
    os.path.join(data_directory, "val"), "NIR")
beta = config.beta

val_RED = ImagesetDataset(
    imset_dir=val_RED_directories, patch_size=128, top_k=None, beta=beta)
val_loader_RED = torch.utils.data.DataLoader(
    val_RED, batch_size=1, shuffle=False, drop_last=False, collate_fn=collateFunction(), pin_memory=True)
val_NIR = ImagesetDataset(
    imset_dir=val_NIR_directories, patch_size=128, top_k=None, beta=beta)
val_loader_NIR = torch.utils.data.DataLoader(
    val_NIR, batch_size=1, shuffle=False, drop_last=False, collate_fn=collateFunction(), pin_memory=True)
# Create model

model = QANet(config)
model.load_state_dict(torch.load('results/saved_models/model_final.pt'))
model = model.cuda()

model.eval()

with torch.no_grad():
    psnr_val_red = []
    scores = []
    for val_step, (lrs, qms, hrs, hr_maps, names) in enumerate(tqdm(val_loader_RED)):
        x_lr = lrs.float().to(config.device)
        x_qm = qms.float().to(config.device)
        x_hr = hrs.float().to(config.device).unsqueeze(1)
        mask = hr_maps.float().to(config.device).unsqueeze(1)
        
        mu_sr = model(x_lr, x_qm)

        mu_sr = np.clip(mu_sr.cpu().detach().numpy()[0].astype(np.float64), 0, 1)
        x_hr = x_hr.cpu().detach().numpy()[0]
        mask = mask.cpu().detach().numpy()[0]

        psnr = shift_cPSNR(mu_sr, x_hr, mask)
        psnr_val_red.append(psnr)
        scores.append(baseline_cpsnrs[names[0]] / psnr)

    print('RED cPSNR: {} Score: {}'.format(np.mean(psnr_val_red), np.mean(scores)))

    psnr_val_nir = []
    scores = []
    for val_step, (lrs, qms, hrs, hr_maps, names) in enumerate(tqdm(val_loader_NIR)):
        x_lr = lrs.float().to(config.device)
        x_qm = qms.float().to(config.device)
        x_hr = hrs.float().to(config.device).unsqueeze(1)
        mask = hr_maps.float().to(config.device).unsqueeze(1)
        
        mu_sr = model(x_lr, x_qm)

        mu_sr = np.clip(mu_sr.cpu().detach().numpy()[0].astype(np.float64), 0, 1)
        x_hr = x_hr.cpu().detach().numpy()[0]
        mask = mask.cpu().detach().numpy()[0]

        psnr = shift_cPSNR(mu_sr, x_hr, mask)
        psnr_val_nir.append(psnr)
        scores.append(baseline_cpsnrs[names[0]] / psnr)

    print('NIR cPSNR: {} Score: {}'.format(np.mean(psnr_val_nir), np.mean(scores)))