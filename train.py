import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from utils.utils import *
from utils.data_loader import *
from utils.evaluator import *
from config import Config
from torchvision.utils import save_image
import warnings
warnings.filterwarnings("ignore")

np.random.seed(216)
torch.manual_seed(216)
torch.cuda.manual_seed_all(216)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', default='results/saved_models', help='Trained model directory')
parser.add_argument('--modules', default=12) # 8, 10, 12
parser.add_argument('--resblocks', default=8) # 6, 8, 10
parser.add_argument('--split', default='all') # all, RED, NIR
parser.add_argument('--sigmoid', default=True) # True, False
parser.add_argument('--arch', default='qanet') # qanet, qanet_noqm, qanet_nowm
parser.add_argument('--scheduler', default=1, type=int) # 1, 2, 3
param = parser.parse_args()

model_time = time.strftime("%Y%m%d_%H%M")
model_name = "model_{}_{}_{}_N{}_M{}".format(model_time, param.split, param.arch, param.modules, param.resblocks)
if param.sigmoid == True:
    model_name += "_S"

# Import config
config = Config()
config.N_modules = int(param.modules)
config.N_resblocks = int(param.resblocks)
config.qem_sigmoid = bool(param.sigmoid)

# Import datasets
data_directory = config.path_prefix
baseline_cpsnrs = readBaselineCPSNR(os.path.join(data_directory, "norm.csv"))
train_set_directories = getImageSetDirectories(
    os.path.join(data_directory, "train"), param.split)

if param.split == 'all':
    split = "RED"
else: 
    split = param.split
val_set_directories = getImageSetDirectories(
    os.path.join(data_directory, "val"), split)

beta = config.beta

train_dataset = ImagesetDataset(imset_dir=train_set_directories, patch_size=config.patch_size, top_k=config.N_lrs, beta=beta)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                           drop_last=False, num_workers=config.workers, collate_fn=collateFunction(), pin_memory=True)

if config.validate:
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
model = model.cuda()

print('No. params: %d' % (sum(p.numel()
      for p in model.parameters() if p.requires_grad),))

# Train
optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-4, weight_decay=1e-5)

if param.scheduler == 1:
    config.N_epoch = 440
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [120, 220, 300, 360, 400], gamma=0.8, last_epoch=-1)
elif param.scheduler == 2:
    config.N_epoch = 460
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [100, 200, 300, 400], gamma=0.6, last_epoch=-1)
elif param.scheduler == 3:
    config.N_epoch = 455
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=56, T_mult=1.5
    )

tot_steps = 0
max_psnr = 0.0

print(model_name)

for epoch in range(config.N_epoch):
    for step, (lrs, qms, hrs, hr_maps, names) in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        x_lr = lrs.float().to(config.device)
        x_qm = qms.float().to(config.device)
        x_hr = hrs.float().to(config.device).unsqueeze(1)
        mask = hr_maps.float().to(config.device).unsqueeze(1)

        mu_sr = model(x_lr, x_qm)
        loss = -get_loss(mu_sr, x_hr, mask)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 15)
        optimizer.step()
    
    tot_steps = tot_steps+step
    scheduler.step()

    if config.validate and epoch % config.val_step == config.val_step-1:
        model.eval()
        with torch.no_grad():
            psnr_val = []
            scores = []
            for val_step, (lrs, qms, hrs, hr_maps, names) in enumerate(val_loader):
                x_lr = lrs.float().to(config.device)
                x_qm = qms.float().to(config.device)
                x_hr = hrs.float().to(config.device).unsqueeze(1)
                mask = hr_maps.float().to(config.device).unsqueeze(1)
                
                mu_sr = model(x_lr, x_qm)

                mu_sr = np.clip(mu_sr.cpu().detach().numpy()[0].astype(np.float64), 0, 1)
                x_hr = x_hr.cpu().detach().numpy()[0]
                mask = mask.cpu().detach().numpy()[0]
            
                psnr = shift_cPSNR(mu_sr, x_hr, mask)
                psnr_val.append(psnr)
                scores.append(baseline_cpsnrs[names[0]] / psnr)

            print('Epoch: {}/{} | {} cPSNR: {} Score: {}'.format(epoch, config.N_epoch, split, np.mean(psnr_val), np.mean(scores)))

        if np.mean(psnr_val) > max_psnr:
            max_psnr = np.mean(psnr_val)
            torch.save(model.state_dict(), os.path.join(param.save_dir, model_name+'_best.pt'))

torch.save(model.state_dict(), os.path.join(param.save_dir, model_name+'_final.pt'))