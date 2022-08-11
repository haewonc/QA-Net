import os
import time
import argparse
import numpy as np
import torch
from utils.utils import *
from utils.data_loader import *
from utils.evaluator import shift_cPSNR, shift_cSSIM
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
parser.add_argument('--band', default='RED', help='Spectral band to validate')
parser.add_argument('--cpkt', default='results/saved_models/red_best.pt', help='Trained model')
param = parser.parse_args()

model_time = time.strftime("%Y%m%d_%H%M")

# Import config
config = Config()

# Import datasets
data_directory = config.path_prefix
baseline_cpsnrs = readBaselineCPSNR(os.path.join(data_directory, "norm.csv"))
val_directories = getImageSetDirectories(
    os.path.join(data_directory, "val"), param.band)
beta = config.beta

val_dataset = ImagesetDataset(
    imset_dir=val_directories, patch_size=128, top_k=config.N_lrs, beta=beta)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False, drop_last=False, collate_fn=collateFunction(), pin_memory=True)

# Create model

model = QANet(config)
model.load_state_dict(torch.load(param.cpkt))
model = model.cuda()

model.eval()

with torch.no_grad():
    psnr_val = []
    ssim_val = []
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
        ssim = shift_cSSIM(mu_sr, x_hr, mask)
        psnr_val.append(psnr)
        ssim_val.append(ssim)
        scores.append(baseline_cpsnrs[names[0]] / psnr)

    print('{} cPSNR: {} cSSIM: {} Score: {}'.format(param.band, np.mean(psnr_val), np.mean(ssim_val), np.mean(scores)))