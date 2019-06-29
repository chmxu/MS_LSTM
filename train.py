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
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', help="model name")
parser.add_argument('--root', help="data root")
parser.add_argument('--pcs', action="store_true", help="predict pcs score")
args = parser.parse_args()
# load the train and test dataset
"""
samples = []
f = open("./data/annotations.txt").readlines()
w = open("./result/final_result.txt", 'w')
for line in f:
    items = line.strip().split(' ')
    samples.append((items[0], float(items[1])))
"""
w = open('./logs/merge_tes_40attn.log', 'w')
loss_log = open('./loss_log/' + args.name + '.txt', 'w')


def train_shuffle(min_mse=200, max_corr=0):
    round_max_spea = 0
    round_min_mse = 200
    # random.shuffle(f)
    # train = samples[:100]
    # test = samples[100:]
    trainset = videoDataset(root=args.root,
                            label="./data/train_dataset.txt", suffix=".npy", transform=transform, data=None, pcs=args.pcs)
    trainLoader = torch.utils.data.DataLoader(trainset,
                                              batch_size=128, shuffle=True, num_workers=0)
    testset = videoDataset(root=args.root,
                           label="./data/test_dataset.txt", suffix='.npy', transform=transform, data=None, pcs=args.pcs)
    testLoader = torch.utils.data.DataLoader(testset,
                                             batch_size=64, shuffle=False, num_workers=0)

    # build the model
    scoring = Scoring(feature_size=4096)
    if torch.cuda.is_available():
        scoring.cuda()  # turn the model into gpu
    # scoring.load_state_dict(torch.load("./models/merge/pcs.pt"))
    total_params = sum(p.numel() for p in scoring.parameters() if p.requires_grad)
    loss_log.write("Total Params: " + str(total_params) + '\n')
    optimizer = optim.Adam(params=scoring.parameters(), lr=0.0005)  # use SGD optimizer to optimize the loss function
    scheduler = lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.7)
    for epoch in range(500):  # total 40 epoches
        # scheduler.step()
        print("Epoch:  " + str(epoch) + "Total Params: %d" % total_params)
        total_regr_loss = 0
        total_sample = 0
        for i, (features, scores) in enumerate(trainLoader):  # get mini-batch
            # print("%d batches have done" % i)
            if torch.cuda.is_available():
                features = Variable(features).cuda()
                scores = Variable(scores).cuda()
            # regression, logits = scoring(features)
            logits, penal = scoring(features)
            if penal is None:
                regr_loss = scoring.loss(logits, scores)
            else:
                regr_loss = scoring.loss(logits, scores) + penal
            # new three lines are back propagation
            optimizer.zero_grad()
            regr_loss.backward()
            # nn.utils.clip_grad_norm(scoring.parameters(), 1.5)
            optimizer.step()
            total_regr_loss += regr_loss.data.item() * scores.shape[0]
            total_sample += scores.shape[0]

        loss_log.write(str(total_regr_loss / total_sample) + '\n')

        print("Classification Loss: " + str(total_regr_loss / total_sample) + '\n')
        # the rest is used to evaluate the model with the test dataset
        torch.save(scoring.state_dict(), './models/epoch{}.pt'.format(epoch))
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
        val_sr, _ = sr(val_truth, val_pred)
        if val_loss / val_sample < min_mse:
            torch.save(scoring.state_dict(), './models/merge/tes_40attn.pt')
        min_mse = min(min_mse, val_loss / val_sample)
        max_corr = max(max_corr, val_sr)
        round_min_mse = min(round_min_mse, val_loss / val_sample)
        round_max_spea = max(val_sr, round_max_spea)
        print("Val Loss: %.2f Correlation: %.2f Min Val Loss: %.2f Max Correlation: %.2f" %
              (val_loss / val_sample, val_sr, min_mse, max_corr))
        scoring.train()
    w.write('MSE: %.2f spearman: %.2f' % (round_min_mse, round_max_spea))
    return min_mse, max_corr


min_mse = 200
max_corr = 0
for time in range(5):
    min_mse, max_corr = train_shuffle(min_mse, max_corr)
w.close()
loss_log.close()
