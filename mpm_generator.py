'''
Motion and Position Map Generator
'''

import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
import hydra
from hydra.utils import to_absolute_path


def compute_vector(cur, nxt, same_count, result_l, result, zeros, z_value, sigma, direction='peak'):
    img_lm = zeros[:, :, 0].copy()  # likelihood image
    img_lm[nxt[1], nxt[0]] = 255
    img_lm = gaussian_filter(img_lm, sigma=sigma, mode='constant')
    img_lm = img_lm / img_lm.max()
    points = np.where(img_lm > 0)
    img = zeros.copy()
    if direction == 'peak':
        for y, x in zip(points[0], points[1]):
            v3d = cur - [x, y]
            v3d = np.append(v3d, z_value)
            v3d = v3d / np.linalg.norm(v3d) * img_lm[y, x]
            img[y, x] = np.array([v3d[1], v3d[0], v3d[2]])

    if direction == 'parallel':
        v3d = cur - nxt
        v3d = np.append(v3d, z_value)
        v3d = v3d / np.linalg.norm(v3d) # normlize offset 
        img[points] = np.array([v3d[1], v3d[0], v3d[2]])
        img = img * img_lm[:, :, None]
    
    if direction == '4d':
        v = cur - nxt
        v = v / np.linalg.norm(np.append(v, z_value))
        img[points] = np.array([])

    img_i = result_l - img_lm
    result[img_i < 0] = img[img_i < 0] # no confict 

    img_i = img_lm.copy() # next cell's map
    img_i[img_i == 0] = 100 # mask out single cell
    img_i = result_l - img_i # if former mask has the same value with this cell 
    if (img_i == 0).any():
        same_count[img_i == 0] += 1 # then tag them with plus 1
    result[img_i == 0] += img[img_i == 0] # accumulate those conflict and save the position with same_count

    result_l = np.maximum(result_l, img_lm)
    return same_count, result_l, result


@hydra.main(config_path='config/mpm_generator.yaml')
def main(cfg):
    track_let = np.loadtxt(to_absolute_path(cfg.file.tracklet)).astype('int')  # 5 columns [frame, id, x, y, parent_id]
    image_size = cv2.imread(to_absolute_path(cfg.file.target)).shape
    save_path = to_absolute_path(cfg.path.save_path)
    os.makedirs(save_path, exist_ok=True)
    z_value = cfg.param.z_value
    sigma = cfg.param.sigma
    itvs = cfg.param.itvs
    direction = cfg.direction

    frames = np.unique(track_let[:, 0])
    ids = np.unique(track_let[:, 1])
    if direction == '4d':
        zeors = np.zeros((image_size[0], image_size[1], 4))
    zeros = np.zeros((image_size[0], image_size[1], 3))
    
    ones = np.ones((image_size[0], image_size[1]))

    for itv in itvs:
        print(f'interval {itv}')
        # z_value = itv # v = a - p 
        save_dir = os.path.join(save_path, f'{itv:03}')
        os.makedirs(save_dir, exist_ok=True)
        output = []

        par_id = -1  # parent id
        for idx, i in enumerate(frames[:-itv]):
            same_count = ones.copy()
            result_lm = zeros[:, :, 0].copy()
            result = zeros.copy()
            bar = tqdm(total=len(ids), position=0)
            for j in ids:
                bar.set_description(f'interval{itv}---frame{i}---id{j}')
                bar.update(1)

                index_current = len(track_let[(track_let[:, 0] == i) & (track_let[:, 1] == j)])
                index_next = len(track_let[(track_let[:, 0] == i + itv) & (track_let[:, 1] == j)])
                if index_next != 0:
                    par_id = track_let[(track_let[:, 0] == i + itv) & (track_let[:, 1] == j)][0, -1]
                if (index_current != 0) & (index_next != 0):
                    data = track_let[(track_let[:, 0] == i) & (track_let[:, 1] == j)][0][2:-1]
                    dnxt = track_let[(track_let[:, 0] == i + itv) & (track_let[:, 1] == j)][0][2:-1]

                    same_count, result_lm, result = compute_vector(data, dnxt, same_count,
                                                                   result_lm,
                                                                   result, zeros, z_value, sigma, direction)

                elif ((index_current == 0) & (index_next != 0) & (par_id != -1)):
                    try:
                        data = track_let[(track_let[:, 0] == i) & (track_let[:, 1] == par_id)][0][2:-1]
                        dnxt = track_let[(track_let[:, 0] == i + itv) & (track_let[:, 1] == j)][0][2:-1]
                        same_count, result_lm, result = compute_vector(data, dnxt, same_count,
                                                                       result_lm,
                                                                       result, zeros, z_value, sigma, direction)
                    except IndexError:
                        print('Error: no parent')
                        print(track_let[(track_let[:, 0] == i + itv) & (track_let[:, 1] == j)][0])
        # import skimage.io as io
        # io.imsave('mask.png', result_lm)
        # breakpoint()
            result = (result / same_count[:, :, None])
            save_name = os.path.join(save_dir, f'{idx:04}.npy')
            np.save(save_name, result.astype('float32'))
    print('finished')


if __name__ == '__main__':
    main()




