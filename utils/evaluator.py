import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import pytorch_ssim
import itertools

def l1_registered_loss(y_true, y_pred, y_mask, HR_SIZE):
    """
    Modified l1 loss to take into account pixel shifts
    """

    y_true = y_true[:,0,:,:]
    y_pred = y_pred[:,0,:,:]
    y_mask = y_mask[:,0,:,:]

    y_shape = y_true.size()
    border = 3
    max_pixels_shifts = 2*border
    size_image = HR_SIZE
    size_croped_image = size_image - max_pixels_shifts
    clear_pixels = size_croped_image*size_croped_image
    cropped_predictions = y_pred[:, border:size_image - border, border:size_image-border]


    X = []
    for i in range(max_pixels_shifts+1):  # range(7)
        for j in range(max_pixels_shifts+1):  # range(7)
            cropped_labels = y_true[:, i:i+(size_image-max_pixels_shifts), j:j+(size_image-max_pixels_shifts)]
            cropped_y_mask = y_mask[:, i:i+(size_image-max_pixels_shifts), j:j+(size_image-max_pixels_shifts)]

            cropped_predictions_masked = cropped_predictions*cropped_y_mask
            cropped_labels_masked = cropped_labels*cropped_y_mask

            total_pixels_masked = torch.sum(cropped_y_mask, dim=(1, 2))

            # bias brightness
            b = (1.0/total_pixels_masked)*torch.sum(cropped_labels_masked-cropped_predictions_masked, dim=(1, 2))
            #print(b.shape)
            b = torch.reshape(b, (y_shape[0], 1, 1))

            corrected_cropped_predictions = cropped_predictions_masked+b
            corrected_cropped_predictions = corrected_cropped_predictions*cropped_y_mask

            if not torch.any(total_pixels_masked==0.):
                l1_loss = torch.sum(torch.abs(cropped_labels_masked-corrected_cropped_predictions), dim=(1,2))/total_pixels_masked
                X.append(l1_loss)
    X = torch.stack(X)
    min_l1 = torch.min(X, dim=0).values

    return torch.mean(min_l1)


def l1_registered_uncertainty_loss(y_true, mu_pred, sigma_pred, y_mask, HR_SIZE):
    """
    Modified l1 loss to take into account pixel shifts
    """

    y_true = y_true[:,0,:,:]
    mu_pred = mu_pred[:,0,:,:]
    sigma_pred = sigma_pred[:,0,:,:]
    y_mask = y_mask[:,0,:,:]

    y_shape = y_true.size()
    border = 3
    max_pixels_shifts = 2*border
    size_image = HR_SIZE
    size_croped_image = size_image - max_pixels_shifts
    clear_pixels = size_croped_image*size_croped_image
    cropped_predictions_mu = mu_pred[:, border:size_image - border, border:size_image-border]
    cropped_predictions_sigma = sigma_pred[:, border:size_image - border, border:size_image-border]


    X = []
    
    for i in range(max_pixels_shifts+1):  # range(7)
        for j in range(max_pixels_shifts+1):  # range(7)
            cropped_labels = y_true[:, i:i+(size_image-max_pixels_shifts), j:j+(size_image-max_pixels_shifts)]
            cropped_y_mask = y_mask[:, i:i+(size_image-max_pixels_shifts), j:j+(size_image-max_pixels_shifts)]

            cropped_predictions_mu_masked = cropped_predictions_mu*cropped_y_mask
            cropped_predictions_sigma_masked = cropped_predictions_sigma*cropped_y_mask
            cropped_labels_masked = cropped_labels*cropped_y_mask

            total_pixels_masked = torch.sum(cropped_y_mask, dim=(1, 2))

            # bias brightness
            b = (1.0/total_pixels_masked)*torch.sum(cropped_labels_masked-cropped_predictions_mu_masked, dim=(1, 2))
            #print(b.shape)
            b = torch.reshape(b, (y_shape[0], 1, 1))

            corrected_cropped_predictions_mu = cropped_predictions_mu_masked+b
            corrected_cropped_predictions_mu = corrected_cropped_predictions_mu*cropped_y_mask
            corrected_cropped_predictions_sigma = cropped_predictions_sigma_masked+b
            corrected_cropped_predictions_sigma = corrected_cropped_predictions_sigma*cropped_y_mask

            #l1_loss = torch.sum(torch.abs(cropped_labels_masked-corrected_cropped_predictions), dim=(1,2))/total_pixels_masked

            y=cropped_labels_masked
            m=corrected_cropped_predictions_mu
            s=corrected_cropped_predictions_sigma
            #l1_loss = torch.sum( torch.log(2*s) + torch.abs(y-m)/s, dim=(1,2))/total_pixels_masked

            if not torch.any(total_pixels_masked==0.):
                l1_loss = torch.sum(s+torch.abs(y-m)*torch.exp(-s), dim=(1,2))/total_pixels_masked
                X.append(l1_loss)
    
    if len(X)>0:
        X = torch.stack(X)
        min_l1 = torch.min(X, dim=0).values
        return torch.mean(min_l1)
    else:
        return None

def NIG_NLL(y, gamma, v, alpha, beta, mask, reduced=True):
    twoBlambda = 2*beta*(1+v)

    nll = 0.5*torch.log(np.pi) - torch.log(v)  \
        - alpha*torch.log(twoBlambda)  \
        + (alpha+0.5) * torch.log(v*(y-gamma)**2 + twoBlambda)  \
        + torch.lgamma(alpha)  \
        - torch.lgamma(alpha+0.5)

    nll = nll*mask

    return torch.mean(nll) if reduce else nll


def evidential_loss(y_true, gamma, v, alpha, beta, reduced=True, coeff=1.0):
    loss_nll = NIG_NLL(y_true, gamma, v, alpha, beta, reduced)
    #loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta)
    #return loss_nll + coeff * loss_reg
    return loss_nll


def registered_evidential_loss(y_true, gamma, v, alpha, beta, y_mask, HR_SIZE):
    """
    Modified l1 loss to take into account pixel shifts
    """

    y_true = y_true[:,0,:,:]
    gamma = gamma[:,0,:,:]
    v = v[:,0,:,:]
    alpha = alpha[:,0,:,:]
    beta = beta[:,0,:,:]
    y_mask = y_mask[:,0,:,:]

    y_shape = y_true.size()
    border = 3
    max_pixels_shifts = 2*border
    size_image = HR_SIZE
    size_croped_image = size_image - max_pixels_shifts
    clear_pixels = size_croped_image*size_croped_image

    cropped_predictions_gamma = gamma[:, border:size_image - border, border:size_image-border]
    cropped_predictions_v = v[:, border:size_image - border, border:size_image-border]
    cropped_predictions_alpha = alpha[:, border:size_image - border, border:size_image-border]
    cropped_predictions_beta = beta[:, border:size_image - border, border:size_image-border]


    X = []
    for i in range(max_pixels_shifts+1):  # range(7)
        for j in range(max_pixels_shifts+1):  # range(7)
            cropped_labels = y_true[:, i:i+(size_image-max_pixels_shifts), j:j+(size_image-max_pixels_shifts)]
            cropped_y_mask = y_mask[:, i:i+(size_image-max_pixels_shifts), j:j+(size_image-max_pixels_shifts)]

            total_pixels_masked = torch.sum(cropped_y_mask, dim=(1, 2))

            cropped_predictions_gamma_masked = cropped_predictions_gamma*cropped_y_mask
            # bias brightness
            b = (1.0/total_pixels_masked)*torch.sum(cropped_labels_masked-cropped_predictions_gamma_masked, dim=(1, 2))
            #print(b.shape)
            b = torch.reshape(b, (y_shape[0], 1, 1))

            corrected_cropped_predictions_gamma = cropped_predictions_gamma+b
            corrected_cropped_predictions_v = cropped_predictions_v+b
            corrected_cropped_predictions_alpha = cropped_predictions_alpha+b
            corrected_cropped_predictions_beta = cropped_predictions_beta+b

            loss = evidential_loss(cropped_labels, corrected_cropped_predictions_gamma, corrected_cropped_predictions_v, corrected_cropped_predictions_alpha, corrected_cropped_predictions_beta, reduced=False)
            loss = loss*cropped_y_mask
            loss = torch.sum(loss, dim=(1,2))/total_pixels_masked

            X.append(loss)
    X = torch.stack(X)
    min_l = torch.min(X, dim=0).values
    return torch.mean(min_l)


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