import numpy as np
import scipy.io as sio
import os
npy_root = '../../skating/c3d_feat/'
f = open("annotations.txt").readlines()
max_feats = []
avr_feats = []
scores = []
for line in f:
    items = line.strip().split(' ')
    scores.append(float(items[1]))
    feat_file = items[0] + '.npy'
    feat = np.load(npy_root + feat_file)
    max_feats.append(np.max(feat, axis=0))
    avr_feats.append(np.mean(feat, axis=0))
max_feats = np.array(max_feats)
avr_feats = np.array(avr_feats)
scores = np.array(scores)
sio.savemat("c3d_max_carl.mat", {"x":max_feats, "y":scores})
sio.savemat("c3d_avr_carl.mat", {"x":avr_feats, "y": scores})

    

