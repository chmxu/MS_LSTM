from dataloader import videoDataset, transform
from model import Scoring
import torch.nn as nn
import torch
import torch.utils.data as data
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
from scipy.stats import spearmanr as sr
import random
#load the train and test dataset
"""
samples = []
f = open("./data/annotations.txt").readlines()
for line in f:
    items = line.strip().split(' ')
    samples.append((items[0], float(items[1])))
"""
w = open("./result/final_result.txt", 'w')
def test_shuffle():
    #random.shuffle(f)
    #train = samples[:100]
    #test = samples[100:]
    testset = videoDataset(root="/home/xuchengming/MM18/figure-skating/c3d_feat",
                   label="./data/test_dataset.txt", suffix='.npy', transform=transform, data=None)
    testLoader = torch.utils.data.DataLoader(testset,
                                      batch_size=64, shuffle=False, num_workers=0)

    #build the model
    scoring = Scoring(feature_size=4096)
    if torch.cuda.is_available():
        scoring.cuda()  #turn the model into gpu
    scoring.load_state_dict(torch.load("./models/merge/pcs.8.35.0.78.pt"))
    scoring.eval()
    min_mse = 200
    max_corr = 0
    for epoch in range(1):  # total 40 epoches
        scoring.eval()
        val_pred = []
        val_sample = 0
        val_loss = 0
        val_truth = []
        for j, (features, scores) in enumerate(testLoader):
            val_truth.append(scores.numpy())
            if torch.cuda.is_available():
                features = Variable(features).cuda()
                scores = Variable(scores).cuda()
            regression, _ = scoring(features)
            val_pred.append(regression.data.cpu().numpy())
            regr_loss = scoring.loss(regression, scores)
            val_loss += (regr_loss.data.item()) * scores.shape[0]
            val_sample += scores.shape[0]
        val_truth = np.concatenate(val_truth)
        val_pred = np.concatenate(val_pred)
    for i in range(val_truth.shape[0]):
        w.write('GT: ' + str(val_truth[i]) + '\t' + "Pred: "+  str(val_pred[i]) + 'Res: ' + str(val_truth[i]-val_pred[i]) + '\n')

for time in range(1):
    test_shuffle()
