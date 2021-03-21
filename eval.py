#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 17:02:23 2021

@author: Junde Xu
"""

import numpy as np
import os

def vmatch(ref, com):  
    lc, lr = len(com), len(ref)
    mat = np.zeros((lc, lr))
    pred_loc = com[:, 2:4]
    ass = []
    for i, cell_loc in enumerate(ref[:, 2:4]):
        dist = np.sqrt(np.sum(np.square(cell_loc[np.newaxis, ::-1] - pred_loc), axis=1))
        cid = np.argmin(dist) if dist[np.argmin(dist)] < 10 else -1
        mat[cid, i] += 1
        ass.append(com[cid, 1])
    return mat 
    

 

def ematch(ref1, com1, ref2, com2, mat1, mat2):
    ed = 0 # edge delete
    ea = 0 # edge add
    ec = 0 # edge change
    
    ctp1 = np.nonzero(np.sum(mat1, axis=1) == 1)[0] # true positive vertex index of compute frame 1
    ctp2 = np.nonzero(np.sum(mat2, axis=1) == 1)[0] # true positive vertex index of compute frame 2
    rtp1 = np.nonzero(mat1[ctp1])[1]
    rtp2 = np.nonzero(mat2[ctp2])[1]
    
    for cidx, ridx in zip(ctp2, rtp2):
        cidc = com2[cidx][1]    # cell id in compute
        pidc = com2[cidx][-1]   # parent id in compute
        cidr = ref2[ridx][1]    # cell id in reference
        pidr = ref2[ridx][-1]
        
        if cidc in com1[ctp1, 1]:
            # compute as track link 
            if cidr not in ref1[rtp1, 1]: 
                if pidr in ref1[rtp1, 1]:
                    ec += 1 
                else:
                    ed += 1
        elif pidc > 0:
            # compute as parent link
            if cidr in ref1[rtp1, 1]:
                ec += 1
            elif pidr not in ref1[rtp1, 1]:
                ed += 1
        else: # nor track link or parent link 
            if cidr in ref1[rtp1, 1] or pidr in ref1[rtp1, 1]:
                ea += 1
            
    return ed, ea, ec

def AOGM(weights, ref1, com1, ref2, com2, mat1, mat2):
    wNS, wFN, wFP, wED, wEA, wEC = weights
    vc = np.sum(mat2, axis=1)
    vr = np.sum(mat2, axis=0)
    ns = np.sum(vc[vc > 1] - 1)
    fn = np.sum(vr == 0)
    fp = np.sum(vc == 0)
    ed, ea, ec = ematch(ref1, com1, ref2, com2, mat1, mat2)
    print("ns: {0}, fn: {1}, fp: {2}, ed: {3}, ea: {4}, ec: {5}".format(ns, fn, fp, ed, ea, ec))
    return wNS*ns + wFN*fn + wFP*fp + wED*ed + wEA*ea + wEC*ec   

def computeperformance(weights, gt_track, pred_track):
    frames_id = np.unique(pred_track[:, 0])
    gt_track = gt_track[gt_track[:, 0] <= np.max(frames_id)]
    
    op = 0
    ref1 = gt_track[gt_track[:, 0] == 0]
    com1 = pred_track[pred_track[:, 0] == 0]
    mat1 = vmatch(ref1, com1)

    for fid in frames_id[1:]:
        ref2 = gt_track[gt_track[:, 0] == fid]
        com2 = pred_track[pred_track[:, 0] == fid]
        mat2 = vmatch(ref2, com2)
        op += AOGM(weights, ref1, com1, ref2, com2, mat1, mat2)
        ref1 = ref2
        com1 = com2
        mat1 = mat2
    print(op)
        

if __name__=='__main__':
    pred_track = np.loadtxt('./output/track/log/PyLog/track.txt')
    gt_track = np.loadtxt('./data/F0017.txt')
    weights = [1, 1, 0, 1, 1, 1]
    computeperformance(weights, gt_track, pred_track)    
    