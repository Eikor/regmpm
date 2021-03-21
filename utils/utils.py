import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from skimage.feature import peak_local_max


def get3chImage(src):
    '''
    Args:
        src: input image
    '''
    chk = src.shape
    if len(chk) == 2:
        out = np.concatenate([src[:, :, None], src[:, :, None], src[:, :, None]], axis=-1)
        return out
    elif chk[-1] == 1:
        out = np.concatenate([src, src, src], axis=-1)
        return out
    else:
        return src


def getImageTable(srcs=[], clm=4, save_name=None):
    '''
    Args:
        srcs: image list
        clm: how many columns
        save_name: save the image table with this name if enter, output it if None
    '''
    white_c = np.full((srcs[0].shape[0], 3, 3), 255).astype('uint8')
    white_r = np.full((3, (srcs[0].shape[1] + 3) * clm - 3, 3), 255).astype('uint8')
    black = np.zeros(srcs[0].shape).astype('uint8')
    out = []

    for i in range(len(srcs)):
        srcs[i] = get3chImage(srcs[i])
        srcs[i] = cv2.hconcat([srcs[i], white_c])
    for i in range(len(srcs) % clm):
        srcs.append(black)

    for l in range(int(len(srcs) / clm)):
        c_imgs = cv2.hconcat(srcs[l * clm:l * clm + clm])
        out.append(c_imgs[:, :-3])
        out.append(white_r)
    out = cv2.vconcat(out)

    if save_name is not None:
        cv2.imwrite(save_name, out[:-3])
    else:
        return out


def RandomFlipper4MPM(seed, input, target):
    if seed == 0:
        return input, target
    elif seed == 1:
        inputH = input[:, :, ::-1].copy()
        targetH = target[:, :, ::-1].copy()
        targetH[1] = -targetH[1]
        return inputH, targetH
    elif seed == 2:
        inputV = input[:, ::-1, :].copy()
        targetV = target[:, ::-1, :].copy()
        targetV[0] = -targetV[0]
        return inputV, targetV
    else:
        inputHV = input[:, ::-1, ::-1].copy()
        targetHV = target[:, ::-1, :: -1].copy()
        targetHV[:2] = -targetHV[:2]
        return inputHV, targetHV

def getSyntheticImage(src1, rate1, src2, rate2, save_name=None):
    src1 = get3chImage(src1)
    src2 = get3chImage(src2)
    out = src1 * rate1 + src2 * rate2
    out[out > 255] = 255
    out = out.astype("uint8")
    if save_name is not None:
        cv2.imwrite(save_name, out)
    else:
        return out
    
def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0])

def tif_2_255(url):
    import os
    import numpy as np
    import skimage.io as io
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    read_url = '/home/siat/sdb/datasets/phc_c2c12/090318/F0017'
    write_url = '/home/siat/sdb/datasets/phc_c2c12/090318/MPM'
    for img_name in tqdm(os.listdir(read_url)):
        rurl_temp = os.path.join(read_url, img_name)
        wurl_temp = os.path.join(write_url, img_name)
        img = io.imread(rurl_temp)
        img = (img / (np.max(img)) * 255).astype('uint8')
        io.imsave(wurl_temp, img, check_contrast=False)

def anno2label():
    # change file name to 0000.tif ... xxxx.tif
    from os import listdir, rename
    from os.path import join
    url = '/home/siat/projects/MICCAI2021/MPM-5908ed8bbcc7c5996287ffe49cc1d79b622cabfc/data_sample/train_imgs/F0018'
    for f in listdir(url):
        old_name = join(url, f)
        new_name = join(url, '{:04d}.tif'.format(int(f[-8:-4]) - 600))
        rename(old_name, new_name)

def getdetection(pred, threshold):
    # lenth of pred as the detection of cell 
    detection = torch.norm(pred, dim=0).cpu().numpy()
    
    return peak_local_max(detection, min_distance=3, threshold_abs=threshold)  

def renormalize(field, z_value):
    '''

    Parameters
    ----------
    field : tensor
        normalized vector with shape of 2*H*W.
    z_value : float.

    Returns
    -------
    

    '''
    a = field[0]
    sign_i = a < 0
    b = field[1]
    sign_j = b < 0
    
    z2 = np.square(np.ones_like(a)*z_value)
    a2 = np.square(a)
    b2 = np.square(b)
    i = np.sqrt(a2 * z2 / (1 - b2 - a2))
    i[sign_i] = -i[sign_i]
    j = np.sqrt(b2 * z2 / (1 - b2 - a2))    
    j[sign_j] = -j[sign_j]
    
    
    

def buildtrack(pred, threshold):
    track = []
    status = []

    detection = getdetection(pred, threshold)
    
    for label in np.unique(split_cell)[1:]:
        frame = []
        i, j = np.where(split_cell == label)
        frame.append([np.mean(i), np.mean(j)])
        track.append(frame)
        status.append(1)
    return track, status
    
def updatetrack(pred, z_value, threshold, track, status):
    detection = getdetection(pred, threshold)
    
    field = pred[:2]
    field[:, detection > threshold] = field[:, detection > threshold] / detection[detection > threshold]
    

    
    # renormalize using z_value

    
    base_cord = np.meshgrid(range(a.shape[0]), range(a.shape[1]), indexing='ij')
    
    _, cells = cv2.threshold(detection, threshold, 255, cv2.THRESH_BINARY)
    ret, split_cell = cv2.connectedComponents(cells.astype('uint8'))

    for label in np.unique(split_cell)[1:]:
        v_i = np.mean(i[split_cell == label])
        v_j = np.mean(j[split_cell == label])
        base_i = np.mean(base_cord[0][split_cell == label])
        base_j = np.mean(base_cord[1][split_cell == label])
        pred_v = np.array([base_i+v_i, base_j+v_j])
        
    
    pass
    
    
    
    
    