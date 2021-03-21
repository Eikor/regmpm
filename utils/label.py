#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 18:58:06 2020

@author: siat
"""

import xml.etree.ElementTree as ET
tree = ET.parse('/home/siat/sdb/datasets/phc_c2c12/090318/anno_human/Human exp1_F0017 Data.xml')
root = tree.getroot()


def write_track(file, cid, ss, pid):
    '''

    Parameters
    ----------
    file : opened file
        file to write.
    cellid: cellid
    ss : xml element
        <fi: first frame>.
    pid : parents id
    Returns
    -------
    None.

    '''
    for t in ss:
        # t: ss's sub element <i: frame, x: , y, >
        file.write('{0} {1} {2} {3} {4}\n'.format(t.get('i'), cid, t.get('y'), t.get('x'), pid))
    

def generation_iter(file, cell, pid):
    cid = cell.get('id')
    tracks = cell.find('ss')
    daughters = cell.find('as')
    if not daughters:
        write_track(file, cid, tracks, pid)
        return
    else:
        write_track(file, cid, tracks, pid)
        for daughter in daughters:
            generation_iter(file, daughter, cid)



file = open('../data/F0017.txt', 'w')
for e1 in root:
    for e2 in e1:
        print(e2.get('name'))
        for e3 in e2:
            for cell in e3:
                print(cell.get('id'))
                generation_iter(file, cell, -1)
                                    
                        