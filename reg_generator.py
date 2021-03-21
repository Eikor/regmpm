#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 13:58:26 2021

@author: Junde Xu
"""

import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
import hydra
from hydra.utils import to_absolute_path


def getxy(track_let, i):
    f = track_let[track_let[:, 0] == i]
    return f[:, 2], f[:, 3]

def getgaussianmap(mag, sigma):
    canvas = np.zeros((100, 100))
    canvas[50, 50] = mag
    canvas = gaussian_filter(canvas, sigma, mode='constant')
    canvas = canvas[canvas>0]
    canvas = canvas / np.max(canvas)
    return canvas.reshape(int(np.sqrt(len(canvas))), -1)
    
def getcellmap(frame, zeros, gaussian_map):
    cmap = zeros[:, :, 0].copy()
    imap = zeros[:, :, 0].copy()
    b = np.array(zeros.shape[0:2])-1
    accumulate_map = zeros[:, :, 0].copy()
    r = ((gaussian_map.shape[0]) - 1)/2
    for cid, xcord, ycord in frame[:, 1:4]:
        tl = np.array([int(ycord - r), int(xcord - r)])
        br = np.array([int(ycord + r), int(xcord + r)])
        shift1 = tl
        shift2 = br
        if any(tl<0):
            shift1 = np.maximum(tl, 0) - tl
            tl = np.maximum(tl, 0)
            cmap[tl[0]:br[0]+1, tl[1]:br[1]+1] = gaussian_map[shift1[0]:, shift1[1]:] 
            imap[tl[0]:br[0]+1, tl[1]:br[1]+1] = cid
        elif any(br - b > 0):
            shift2 = (2*r+1 + np.minimum(br, b) - br).astype('int')
            br = np.minimum(br, b)
            cmap[tl[0]:br[0]+1, tl[1]:br[1]+1] = gaussian_map[:shift2[0], :shift2[1]]
            imap[tl[0]:br[0]+1, tl[1]:br[1]+1] = cid
        else:
            cmap[tl[0]:br[0]+1, tl[1]:br[1]+1] = gaussian_map
            imap[tl[0]:br[0]+1, tl[1]:br[1]+1] = cid
        accumulate_map = np.maximum(cmap, accumulate_map)        
    return accumulate_map, imap

def computevector(pf, cf, cam, cim, zeros, z_value, norm): # previous frame, current frame, cell id map
    transvec = zeros[:, :, :3].copy() # h*w*3
    if norm:
        transvec[:, :, 2] = z_value
    vec = [0, 0] 
    for cid, x, y, p in cf[:, 1:]:
        if cid in pf[:, 1]:
            # track link 
            vec = np.squeeze(pf[pf[:, 1] == cid, 2:4] - [x, y])
        elif p in pf[:, 1]:
            # parent link
            vec = np.squeeze(pf[pf[:, 1] == p, 2:4] - [x, y])
        transvec[cim == cid, 0:-1] = vec[::-1]
    if norm:
        transvec = transvec / np.linalg.norm(transvec, axis=2, keepdims=True)
        transvec = transvec * cam[:, :, None]
    else:
        transvec[:, :, -1] = cam
    return transvec
        
    

def main(norm=True, pre=True, dataset='F0018'):
    track_let = np.loadtxt(to_absolute_path('annotation/090318-C2C12P7_{}.txt'.format(dataset))).astype('int')  # 5 columns [frame, id, x, y, parent_id]
    image_size = cv2.imread(to_absolute_path('data/train_imgs/F0018/0000.tif')).shape
    if norm:
        if pre:
            save_path = to_absolute_path('data/train_mpms/{}_norm_pre'.format(dataset))
        else:
            save_path = to_absolute_path('data/train_mpms/{}_norm'.format(dataset))
    elif pre:
        save_path = to_absolute_path('data/train_mpms/{}_pre'.format(dataset))
    else:
        save_path = to_absolute_path('data/train_mpms/{}'.format(dataset))
   
    os.makedirs(save_path, exist_ok=True)
    z_value = 5 # cfg.param.z_value
    sigma = 5 # cfg.param.sigma
    itvs = [1, 3, 5, 7, 9]    # cfg.param.itvs
    direction = 'parallel' # cfg.direction

    frames = np.unique(track_let[:, 0])
    gaussian_map = getgaussianmap(255, sigma)

    zeros = np.zeros((image_size[0], image_size[1], 4 if pre else 3))
    ones = np.ones((image_size[0], image_size[1]))
    
    for itv in itvs:
        print(f'interval {itv}')
        # z_value = itv # v = a - p 
        save_dir = os.path.join(save_path, f'{itv:03}')
        os.makedirs(save_dir, exist_ok=True)
        output = []
        par_id = -1  # parent id
        
        pf = track_let[track_let[:, 0] == frames[0]] #previous frame (first frame)
        if pre:
            pam, pim = getcellmap(pf, zeros, gaussian_map) # previous cell activation/id map
        bar = tqdm(total=len(frames)-itv-1, position=0)
        for idx, i in enumerate(frames[itv:]):
            bar.update(1)
            cf = track_let[track_let[:, 0] == i] # current frame
            cam, cim = getcellmap(cf, zeros, gaussian_map)
            ###### assign cell activation map #####
            result = zeros.copy()
            if pre:
                result[:, :, 0] = pam
            
            ###### assign transportation vector #####
            result[:, :, (1 if pre else 0):] = computevector(pf, cf, cam, cim, zeros, z_value, norm)
            # result = computevector(pf, cf, cam, cim, zeros, z_value)
            save_name = os.path.join(save_dir, f'{idx:04}.npy')
            np.save(save_name, result.astype('float32'))
            if pre:
                pam, pim = cam, cim
    print('finish')


if __name__ == '__main__':
    main(norm=True, pre=True, dataset='F0018')
    main(norm=True, pre=True, dataset='F0017')
    main(norm=False, pre=True, dataset='F0018')
    main(norm=False, pre=True, dataset='F0017')
    main(norm=False, pre=False, dataset='F0018')
    main(norm=False, pre=False, dataset='F0017')
