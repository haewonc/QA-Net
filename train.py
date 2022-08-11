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
train_set_directories = getImageSetDirectories(
    os.path.join(data_directory, "train"))
val_RED_directories = getImageSetDirectories(
    os.path.join(data_directory, "val"), "RED")

beta = config.beta

train_dataset = ImagesetDataset(imset_dir=train_set_directories, patch_size=config.patch_size, top_k=config.N_lrs, beta=beta)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                           drop_last=False, num_workers=config.workers, collate_fn=collateFunction(), pin_memory=True)

if config.validate:
    val_RED = ImagesetDataset(
        imset_dir=val_RED_directories, patch_size=128, top_k=config.N_lrs, beta=beta)
    val_loader_RED = torch.utils.data.DataLoader(
        val_RED, batch_size=1, shuffle=False, drop_last=False, collate_fn=collateFunction(), pin_memory=True)

# Create model

model = QANet(config)
model = model.cuda()

print('No. params: %d' % (sum(p.numel()
      for p in model.parameters() if p.requires_grad),))

# Train
optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, [120, 220, 300, 360, 400, 440, 480], gamma=0.8, last_epoch=-1)

tot_steps = 0
max_psnr = 0.0

print(model_time)

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

        if step % 100 == 99:
            save_image(mu_sr[0], "results/train_imgs/{}_SR.png".format(step+1))
            save_image(x_hr[0], "results/train_imgs/{}_HR.png".format(step+1))
    
    tot_steps = tot_steps+step
    scheduler.step()

    if config.validate and epoch % config.val_step == config.val_step-1:
        model.eval()
        with torch.no_grad():
            psnr_val_red = []
            scores = []
            for val_step, (lrs, qms, hrs, hr_maps, names) in enumerate(val_loader_RED):
                x_lr = lrs.float().to(config.device)
                x_qm = qms.float().to(config.device)
                x_hr = hrs.float().to(config.device).unsqueeze(1)
                mask = hr_maps.float().to(config.device).unsqueeze(1)
                
                mu_sr = model(x_lr, x_qm)
                
                if val_step % 10 == 9:
                    save_image(mu_sr, "results/val_imgs/RED_{}_SR.png".format(val_step+1))
                    save_image(x_hr, "results/val_imgs/RED_{}_HR.png".format(val_step+1))

                mu_sr = np.clip(mu_sr.cpu().detach().numpy()[0].astype(np.float64), 0, 1)
                x_hr = x_hr.cpu().detach().numpy()[0]
                mask = mask.cpu().detach().numpy()[0]
            
                psnr = shift_cPSNR(mu_sr, x_hr, mask)
                psnr_val_red.append(psnr)
                scores.append(baseline_cpsnrs[names[0]] / psnr)

            print('Epoch: {}/{} | RED cPSNR: {} Score: {}'.format(epoch, config.N_epoch, np.mean(psnr_val_red), np.mean(scores)))
         
torch.save(model.state_dict(), os.path.join(param.save_dir,'model_final_'+model_time+'.pt'))