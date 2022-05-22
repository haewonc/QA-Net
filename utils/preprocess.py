from utils import load_dataset
import numpy as np
import os
import random 
import argparse 
import torch 
import torch.nn.functional as F
from tqdm import tqdm
import heapq
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--ddir", help="path of the dataset", default='../../probav_data/')
args = parser.parse_args()

dataset_dir = args.ddir
ITER = 100
L = 9

# validation set loading
X_RED_val, X_RED_val_masks, y_RED_val, y_RED_val_masks, RED_val_names = load_dataset(base_dir=dataset_dir, part="val", band="RED", W=128)
X_NIR_val, X_NIR_val_masks, y_NIR_val, y_NIR_val_masks, NIR_val_names = load_dataset(base_dir=dataset_dir, part="val", band="NIR", W=128)

print(f"Val RED scenes: {len(X_RED_val)} | Val RED y shape: {y_RED_val.shape}")
print(f"Val NIR scenes: {len(X_NIR_val)} | Val NIR y shape: {y_NIR_val.shape}")

# test set loading
X_RED_test, X_RED_test_masks, y_RED_test_masks, RED_test_names = load_dataset(base_dir=dataset_dir,part="test",band="RED",  W=128)
X_NIR_test, X_NIR_test_masks, y_NIR_test_masks,  NIR_test_names = load_dataset(base_dir=dataset_dir,part="test",band="NIR",  W=128)

print(f"Test RED scenes: {len(X_RED_test)}")
print(f"Test NIR scenes: {len(X_NIR_test)}")

for i, name in enumerate(tqdm(RED_val_names)):
    set_dir = dataset_dir+"val/RED/{}/".format(name)
    T = X_RED_val[i].shape[-1]
    hqm = torch.tensor(y_RED_val_masks[i], dtype=torch.int).squeeze(2).unsqueeze(0).unsqueeze(0)
    shqm = F.conv2d(hqm, weight=torch.ones((1, 1, 3, 3), dtype=torch.int), stride=3).squeeze(0).squeeze(0).numpy().reshape(-1)
    max_qm_ids = []
    min_qm = -1
    for it in range(ITER):
      indices = random.sample(range(T), L)
      a = np.sum(X_RED_val_masks[i][:, :, indices], axis=2).reshape(-1)
      a = np.delete(a, np.where(shqm==0)[0])
      min_it = sum(a[heapq.nsmallest(10, range(len(a)), a.take)])
      if min_it > min_qm:
        min_qm = min_it 
        max_qm_ids = indices
    print(min_qm)
    for j in range(T):
      if j not in max_qm_ids:
        os.remove(set_dir+"LR{0:03}.png".format(j))
        os.remove(set_dir+"QM{0:03}.png".format(j))

for i, name in enumerate(tqdm(NIR_val_names)):
    set_dir = dataset_dir+"val/NIR/{}/".format(name)
    T = X_NIR_val[i].shape[-1]
    hqm = torch.tensor(y_NIR_val_masks[i], dtype=torch.int).squeeze(2).unsqueeze(0).unsqueeze(0)
    shqm = F.conv2d(hqm, weight=torch.ones((1, 1, 3, 3), dtype=torch.int), stride=3).squeeze(0).squeeze(0).numpy().reshape(-1)
    max_qm_ids = []
    min_qm = -1
    for it in range(ITER):
      indices = random.sample(range(T), L)
      a = np.sum(X_NIR_val_masks[i][:, :, indices], axis=2).reshape(-1)
      a = np.delete(a, np.where(shqm==0)[0])
      min_it = sum(a[heapq.nsmallest(10, range(len(a)), a.take)])
      if min_it > min_qm:
        min_qm = min_it 
        max_qm_ids = indices
    print(min_qm)
    for j in range(T):
      if j not in max_qm_ids:
        os.remove(set_dir+"LR{0:03}.png".format(j))
        os.remove(set_dir+"QM{0:03}.png".format(j))

for i, name in enumerate(tqdm(RED_test_names)):
    set_dir = dataset_dir+"test/RED/{}/".format(name)
    T = X_RED_test[i].shape[-1]
    hqm = torch.tensor(y_RED_test_masks[i], dtype=torch.int).squeeze(2).unsqueeze(0).unsqueeze(0)
    shqm = F.conv2d(hqm, weight=torch.ones((1, 1, 3, 3), dtype=torch.int), stride=3).squeeze(0).squeeze(0).numpy().reshape(-1)
    max_qm_ids = []
    min_qm = -1
    for it in range(ITER):
      indices = random.sample(range(T), L)
      a = np.sum(X_RED_test_masks[i][:, :, indices], axis=2).reshape(-1)
      a = np.delete(a, np.where(shqm==0)[0])
      min_it = sum(a[heapq.nsmallest(10, range(len(a)), a.take)])
      if min_it > min_qm:
        min_qm = min_it 
        max_qm_ids = indices
    print(min_qm)
    for j in range(T):
      if j not in max_qm_ids:
        os.remove(set_dir+"LR{0:03}.png".format(j))
        os.remove(set_dir+"QM{0:03}.png".format(j))

for i, name in enumerate(tqdm(NIR_test_names)):
    set_dir = dataset_dir+"test/NIR/{}/".format(name)
    T = X_NIR_test[i].shape[-1]
    hqm = torch.tensor(y_NIR_test_masks[i], dtype=torch.int).squeeze(2).unsqueeze(0).unsqueeze(0)
    shqm = F.conv2d(hqm, weight=torch.ones((1, 1, 3, 3), dtype=torch.int), stride=3).squeeze(0).squeeze(0).numpy().reshape(-1)
    max_qm_ids = []
    min_qm = -1
    for it in range(ITER):
      indices = random.sample(range(T), L)
      a = np.sum(X_NIR_test_masks[i][:, :, indices], axis=2).reshape(-1)
      a = np.delete(a, np.where(shqm==0)[0])
      min_it = sum(a[heapq.nsmallest(10, range(len(a)), a.take)])
      if min_it > min_qm:
        min_qm = min_it 
        max_qm_ids = indices
    print(min_qm)
    for j in range(T):
      if j not in max_qm_ids:
        os.remove(set_dir+"LR{0:03}.png".format(j))
        os.remove(set_dir+"QM{0:03}.png".format(j))