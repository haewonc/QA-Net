'''
https://github.com/diegovalsesia/piunet
https://arxiv.org/abs/2105.12409
'''

import cv2
import numpy as np
from glob import glob
from scipy.ndimage import shift
from skimage.transform import rescale
from skimage.feature import masked_register_translation
from tqdm import tqdm
import csv
import os
import torch

def load_dataset(base_dir, part, band, W):
    """
    Load the original proba-v dataset already splitted in train, validation and test
    
    Parameters
    ----------
    base_dir: str
        path to the original dataset folder
    part: str
        'train', 'val' or test string
    band: str
        string with the band 'NIR' or 'RED'
    """
    imgsets = sorted(glob(base_dir+"/"+part+"/"+band+"/*"))
    
    X = []; X_masks = []; y = []; y_masks = []; names = []
    for imgset in tqdm(imgsets):
        LRs = sorted(glob(imgset+"/LR*.png"))
        QMs = sorted(glob(imgset+"/QM*.png"))
        T = len(LRs)
        
        LR = np.empty((W,W,T),dtype="uint16")
        QM = np.empty((W,W,T),dtype="bool")
        
        for i,img in enumerate(LRs):
            LR[...,i] = cv2.imread(img,cv2.IMREAD_UNCHANGED)
        for i,img in enumerate(QMs):
            QM[...,i] = cv2.imread(img,cv2.IMREAD_UNCHANGED).astype("bool")
        
        X.append(LR)
        X_masks.append(QM)
        names.append(imgset.split('/')[-1])
        
        if part != "test":
            y.append(cv2.imread(imgset+"/HR.png",cv2.IMREAD_UNCHANGED)[...,None])
        y_masks.append(cv2.imread(imgset+"/SM.png",cv2.IMREAD_UNCHANGED).astype("bool")[...,None])
    
    if part != "test":
        return X,X_masks,np.array(y),np.array(y_masks), names
    else:
         return X,X_masks,y_masks,names
    
def register_dataset(X, masks):
    """
    Register the input tensor X of shape (B, H, W, T) with respect to the image with the best quality map
    
    Parameters
    ----------
    X: numpy array
        tensor X to register
    masks: numpy array
        tensor with the quality maps of X
    """
    X_reg = []
    masks_reg = []
    
    for i in tqdm(range(len(X))):
        img_reg,m_reg = register_imgset(X[i], masks[i])
        X_reg.append(img_reg)
        masks_reg.append(m_reg)
    
    return X_reg,masks_reg


def register_imgset(imgset, mask):
    """
    Register the input tensor imgset of shape (H, W, T) with respect to the image with the best quality map
    
    Parameters
    ----------
    imgset: numpy array
        imgset to register
    masks: numpy array
        tensor with the quality maps of the imgset
    """
    ref = imgset[...,np.argmax(np.mean(mask,axis=(0,1)))] #best image
    imgset_reg = np.empty(imgset.shape)
    mask_reg = np.empty(mask.shape)
    
    for i in range(imgset.shape[-1]):
        x = imgset[...,i]; m = mask[...,i]
        s = masked_register_translation(ref, x, m)
        x = shift(x, s, mode='reflect')
        m = shift(m, s, mode='constant', cval=0)
        imgset_reg[...,i] = x
        mask_reg[...,i] = m
        
    return imgset,mask_reg


    
def select_T_images(X, masks, T=9, thr=0.85, remove_bad=True):
    """
    Select the best T images of each imgset in X
    
    Parameters
    ----------
    X: numpy array
        tensor X with all scenes
    masks: numpy array
        tensor with the quality maps of all imgset in X
    T: int
        number of temporal steps to select
    thr: float
        percentage for the quality check
    remove_bad: bool
        remove bad timesteps
    """
    X_selected = []
    masks_selected = []
    remove_indexes = []
    
    for i in tqdm(range(len(X))):
        imgset = X[i]; m = masks[i]
        clearance = np.mean(m,axis=(0,1))
        clear_imgset = imgset[...,clearance > thr]
        clear_m = m[...,clearance > thr]
        clearance = clearance[clearance > thr]
        if not len(clearance):
            if remove_bad: #imgset all under threshold, removed
                print("Removing number",i)
                remove_indexes.append(i)
                continue     
            else: # for testing images, take best image
                best_index = np.argmax(np.mean(m,axis=(0,1)))
                clearance = np.mean(m,axis=(0,1))[best_index:best_index+1]
                clear_imgset = imgset[...,best_index:best_index+1]
                clear_m = m[...,best_index:best_index+1]
                
        
        sorted_clearances_indexes = list(np.argsort(clearance)[::-1])    #sort decrescent
        delta = T - len(clearance)

        if delta>0:  # repeat random indexes because we have less than T
            random_indexes = []
            for _ in range(delta):
                random_indexes.append(np.random.choice(sorted_clearances_indexes))
            sorted_clearances_indexes += random_indexes
        
        X_selected.append(clear_imgset[...,sorted_clearances_indexes[:T]])   #take T images        
        masks_selected.append(clear_m[...,sorted_clearances_indexes[:T]])    #take T masks  
            
    return np.array(X_selected), remove_indexes

def sub_images(X,d,s,n): 
    """
    Generate patches util
    """
    l = n**2
    ch = X.shape[-1]
    k = np.empty((l,d,d,ch))
    
    for i in range(n):
        for j in range(n):
            sub = X[i*s:i*s+d,j*s:j*s+d]
            k[n*i+j] = sub
    return k


def gen_sub(array,d,s,verbose=True):
    """
    Generate patches 
    
    Parameters
    ----------
    array: numpy array
        tensor X with all scenes
    d: int
        dimension of the pathches
    s: int
        stride between pathces
    verbose: bool
        print output info
    """
    if len(array.shape) != 4: raise ValueError("Wrong array shape.")
    
    l = len(array)
    d_o = array.shape[1]
    ch = array.shape[-1]
    d = int(d)
    s = int(s)
    
    n = (d_o - d)/s + 1
    if int(n) != n: raise ValueError("d, s and n should be integer values.")
    
    n = int(n)
    
    X_sub = np.empty((l*(n**2),d,d,ch))
    for i,X in enumerate(array):    
        sub = sub_images(X,d,s,n)
        X_sub[i*n**2:(i+1)*n**2] = sub

    if verbose:
        print(X_sub.shape)
    return X_sub


def bicubic(X, scale = 3):
    """
    Rescale with bicubic operation
    
    Parameters
    ----------
    X: numpy array
        tensor X to upscale
    scale: int
        scale dimension
    """
    if len(X.shape) == 3:
        X = np.expand_dims(X,axis=0)
    if len(X.shape) != 4: raise ValueError("Wrong array shape.")
    shape = [X.shape[0],X.shape[1]*scale,X.shape[2]*scale,X.shape[-1]]
    
    X_upscaled = np.empty(shape)
    
    for i,lr in enumerate(X):
        sr_img = rescale(lr,scale=scale,order=3,mode='edge',
                     anti_aliasing=False, multichannel=True, preserve_range=True) #bicubic
        X_upscaled[i] = sr_img

    return X_upscaled

'''
Reference
https://github.com/ElementAI/HighRes-net
https://arxiv.org/abs/2002.06460
'''

def readBaselineCPSNR(path):
    """
    Reads the baseline cPSNR scores from `path`.
    Args:
        filePath: str, path/filename of the baseline cPSNR scores
    Returns:
        scores: dict, of {'imagexxx' (str): score (float)}
    """
    scores = dict()
    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            scores[row[0].strip()] = float(row[1].strip())
    return scores


def getImageSetDirectories(data_dir, channels="all"):
    """
    Returns a list of paths to directories, one for every imageset in `data_dir`.
    Args:
        data_dir: str, path/dir of the dataset
    Returns:
        imageset_dirs: list of str, imageset directories
    """
    if channels=="all":
        channel_dirs = ['RED', 'NIR']
    else:
        channel_dirs = []
        channel_dirs.append(channels)
        
    imageset_dirs = []
    for channel_dir in channel_dirs:
        path = os.path.join(data_dir, channel_dir)
        for imageset_name in os.listdir(path):
            imageset_dirs.append(os.path.join(path, imageset_name))
    return imageset_dirs


    
class collateFunction():
    """ Util class to create padded batches of data. """

    def __call__(self, batch):
        return self.collateFunction(batch)

    def collateFunction(self, batch):
        """
        Custom collate function to adjust a variable number of low-res images.
        Args:
            batch: list of imageset
        Returns:
            padded_lr_batch: tensor (B, min_L, W, H), low resolution images
            alpha_batch: tensor (B, min_L), low resolution indicator (0 if padded view, 1 otherwise)
            hr_batch: tensor (B, W, H), high resolution images
            hm_batch: tensor (B, W, H), high resolution status maps
            isn_batch: list of imageset names
        """
        
        lr_batch = []  # batch of low-resolution views
        qm_batch = []  # batch of indicators (0 if padded view, 1 if genuine view)
        hr_batch = []  # batch of high-resolution views
        hm_batch = []  # batch of high-resolution status maps
        isn_batch = []  # batch of site names

        train_batch = True

        for imageset in batch:

            lrs = imageset['lr']
            qms = imageset['qm']

            L, H, W = lrs.shape

            pad = torch.zeros(0, H, W)
            lr_batch.append(torch.cat([lrs, pad], dim=0))
            qm_batch.append(torch.cat([qms, pad], dim=0))

            hr = imageset['hr']
            if train_batch and hr is not None:
                hr_batch.append(hr)
            else:
                train_batch = False

            hm_batch.append(imageset['hr_map'])
            isn_batch.append(imageset['name'])

        padded_lr_batch = torch.stack(lr_batch, dim=0)
        padded_qm_batch = torch.stack(qm_batch, dim=0)

        if train_batch:
            hr_batch = torch.stack(hr_batch, dim=0)
            hm_batch = torch.stack(hm_batch, dim=0)

        return padded_lr_batch, padded_qm_batch, hr_batch, hm_batch, isn_batch