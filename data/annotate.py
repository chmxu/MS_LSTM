import json
import os
import argparse
import csv
import numpy as np
from copy import deepcopy
train_idx = np.random.choice(500, 400, replace=False)
test_idx = [item for item in range(500) if item not in train_idx]
annotations = open("fs_dataset.csv")
anno = csv.DictReader(annotations)
train = open("train_dataset.txt", 'w')
test = open("test_dataset.txt", 'w')
whole_list = []
for row in anno:
    line = ' '.join([row['number'], row['tes'], row['pcs'], row['ded']]) + '\n'
    if int(row['number']) in train_idx:
        train.write(line)
    else:
        test.write(line)
train.close()
test.close()

