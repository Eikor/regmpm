import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import cv2
from os import listdir, path
from glob import glob
from MPM_Net import MPMNet

def eval_net(net, loader, device, criterion, writer, global_step):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    # mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    total_error = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, mpms_gt = batch['img'], batch['mpm']
            imgs = imgs.to(device=device, dtype=torch.float32)
            mpms_gt = mpms_gt.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                mpms_pred = net(imgs)

            error = criterion(mpms_pred, mpms_gt)

            pbar.update()

        # print(imgs.shape)
        # writer.add_images('images/1', imgs[:, :1], global_step)
        # writer.add_images('images/2', imgs[:, 1:], global_step)

        # writer.add_images('mpms/true', mpms_gt, global_step)
        # writer.add_images('mpms/pred', mpms_pred, global_step)

        # mags_gt = mpms_gt.pow(2).sum(dim=1, keepdim=True).sqrt()
        # mags_pred = mpms_pred.pow(2).sum(dim=1, keepdim=True).sqrt()

        # writer.add_images('mags/true', mags_gt, global_step)
        # writer.add_images('mags/pred', mags_pred, global_step)

    net.train()
    return error

def tracking(net, device, file, images_url, cover):
    images = sorted(glob(path.join(images_url, '*.tif')))
    h, w = cover.shape
    threshold = 1
    z_value = 5
    from utils.utils import buildtrack, updatetrack

    '''
    tracks:
        - frames
            - cell associate with same label
    '''
    
    for idx in range(len(images)):
        frame = []
        i1 = cv2.imread(images[idx], -1)
        i2 = cv2.imread(images[idx+1], -1)
        X = torch.from_numpy(np.concatenate([i1[None, :], i2[None, :]], axis=0)).type(torch.FloatTensor).to(device)
        with torch.no_grad():
            pred = net(X.unsqueeze(0)).squeeze(0)
        
        '''
            pred: 
                lenth is the cell probability
                the first 2 channel is the transfer direction which normalized with z value 
                
        ''' 
        
        if idx == 0: # build track
            track, status = buildtrack(pred, threshold)
        
        track, status = updatetrack(field, detection, z_value, threshold, track, status)
        
        
        
        # for cell_id in np.unique(labels)[1:]: # reassign label with former label
        #     center = np.mean(infered_mask[:, labels==cell_id], axis=1).astype('int') # get previous center
        #     cid = cover[center[0], center[1]]
        #     if cid != 0: # if not background
        #         pid = cid
        #         labels[cover_next == cell_id] = cid
        #         file.write('{0} {1} {2} {3} {4}\n'.format(idx, cid, center[0], center[1], pid))
        #     else: # tracking cell from cover, no migration in 
        #         cover_next[cover_next == cell_id] = 0
        # cover = (cover_next>0) * labels
        
        
        
if __name__ == '__main__':
    file = open('data/tracking.txt', 'w')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = MPMNet(n_channels=2, n_classes=3, bilinear=True)    
    net.load_state_dict(
            torch.load('outputs/2020-12-25/16-55-20/checkpoints/CP_epoch300.pth', map_location=device)
        )
    net.to(device)
    
    images_url = '/home/siat/sdb/datasets/phc_c2c12/090318/MPM'
    gt_track = np.loadtxt('data/F0017.txt').astype('int')
    
    cell = gt_track[gt_track[:, 0] == 0]
    h, w = cv2.imread(path.join(images_url, listdir(images_url)[0]), -1).shape
     
    cover = np.zeros([h, w])    
    for c in cell:
        cover[(c[2] - 12):(c[2]+12), (c[3] - 12) :(c[3]+12)] = c[1]
        file.write('{0} {1} {2} {3} {4}\n'.format(c[0], c[1], c[2], c[3], c[4]))
    tracking(net, device, file, images_url, cover)
    file.close()  
        
        
        
        
        
        
        
        
        
        
        
        
        