import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from skimage.metrics import structural_similarity as ssim
import itertools


def shift_cSSIM(sr, hr, hr_map):
    size_image = 384
    border = 3
    max_pixels_shifts = 2*border
    cropped_predictions = sr[:, border:size_image - border, border:size_image-border]

    X = []
    for i in range(max_pixels_shifts+1):  # range(7)
        for j in range(max_pixels_shifts+1):  # range(7)
            cropped_labels = hr[:, i:i+(size_image-max_pixels_shifts), j:j+(size_image-max_pixels_shifts)]
            cropped_y_mask = hr_map[:, i:i+(size_image-max_pixels_shifts), j:j+(size_image-max_pixels_shifts)]

            cropped_predictions_masked = cropped_predictions * cropped_y_mask
            cropped_labels_masked = cropped_labels * cropped_y_mask

            total_pixels_masked = np.sum(cropped_y_mask, axis=(1, 2))

            # bias brightness
            b = (1.0/total_pixels_masked) * np.sum(cropped_labels_masked-cropped_predictions_masked, axis=(1, 2))
            b = np.reshape(b, (1, 1, 1))

            corrected_cropped_predictions = cropped_predictions_masked + b
            corrected_cropped_predictions = corrected_cropped_predictions * cropped_y_mask
            
            cssim = ssim(cropped_labels_masked[0], corrected_cropped_predictions[0], win_size=11)

            X.append(cssim)

    return max(X)

def cPSNR(sr, hr, hr_map):
    """
    Clear Peak Signal-to-Noise Ratio. The PSNR score, adjusted for brightness and other volatile features, e.g. clouds.
    Args:
        sr: numpy.ndarray (n, m), super-resolved image
        hr: numpy.ndarray (n, m), high-res ground-truth image
        hr_map: numpy.ndarray (n, m), status map of high-res image, indicating clear pixels by a value of 1
    Returns:
        cPSNR: float, score
    """

    if len(sr.shape) == 2:
        sr = sr[None, ]
        hr = hr[None, ]
        hr_map = hr_map[None, ]

    if sr.dtype.type is np.uint16:  # integer array is in the range [0, 65536]
        sr = sr / np.iinfo(np.uint16).max  # normalize in the range [0, 1]
    else:
        assert 0 <= sr.min() and sr.max() <= 1, 'sr.dtype must be either uint16 (range 0-65536) or float64 in (0, 1).'
    if hr.dtype.type is np.uint16:
        hr = hr / np.iinfo(np.uint16).max

    n_clear = np.sum(hr_map, axis=(1, 2))  # number of clear pixels in the high-res patch
    diff = hr - sr
    bias = np.sum(diff * hr_map, axis=(1, 2)) / n_clear  # brightness bias
    cMSE = np.sum(np.square((diff - bias[:, None, None]) * hr_map), axis=(1, 2)) / n_clear
    cPSNR = -10 * np.log10(cMSE)  # + 1e-10)

    if cPSNR.shape[0] == 1:
        cPSNR = cPSNR[0]

    return cPSNR


def patch_iterator(img, positions, size):
    """Iterator across square patches of `img` located in `positions`."""
    for x, y in positions:
        yield get_patch(img=img, x=x, y=y, size=size)


def get_patch(img, x, y, size=32):
    patch = img[..., x:(x + size), y:(y + size)] 
    return patch

def shift_cPSNR(sr, hr, hr_map, border_w=3):
    size = sr.shape[1] - (2 * border_w)  # patch size
    sr = get_patch(img=sr, x=border_w, y=border_w, size=size)
    pos = list(itertools.product(range(2 * border_w + 1), range(2 * border_w + 1)))
    iter_hr = patch_iterator(img=hr, positions=pos, size=size)
    iter_hr_map = patch_iterator(img=hr_map, positions=pos, size=size)
    site_cPSNR = np.array([cPSNR(sr, hr, hr_map) for hr, hr_map in zip(iter_hr, iter_hr_map)])
    max_cPSNR = np.max(site_cPSNR, axis=0)
    return max_cPSNR

def get_loss(srs, hrs, hr_maps):
    criterion = nn.MSELoss(reduction='none')
    nclear = torch.sum(hr_maps, dim=(1, 2)) + 1 # Number of clear pixels in target image
    bright = torch.sum(hr_maps * (hrs - srs), dim=(1, 2)).clone().detach() 
    bright = bright / nclear  # Correct for brightness
    loss = torch.sum(hr_maps * criterion(srs + bright.view(-1, 1, 1), hrs), dim=(1, 2)) / nclear  # cMSE(A,B) for each point

    one = torch.ones(loss.size()).cuda()
    loss = torch.where(nclear>1, loss, one)
    loss = -10 * torch.log10(loss)
    loss = torch.sum(loss, dim=1)/torch.sum(nclear>1, dim=1)
    loss = torch.mean(loss)

    return loss