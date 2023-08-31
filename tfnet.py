from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils import data
import itertools
import re
import random
import time
import sys
from os.path import exists
from tqdm.auto import trange

sys.path.append("TF-net")
from model import LES
from torch.autograd import Variable
from penalty import DivergenceLoss
from train import Dataset, train_epoch, eval_epoch, test_epoch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings("ignore")

#best_params: kernel_size 3, learning_rate 0.001, dropout_rate 0, batch_size 120, input_length 25, output_length 4
min_mse = 1
time_range  = 6
output_length = 4
input_length = 26
learning_rate = 0.001
dropout_rate = 0
kernel_size = 3
batch_size = 32

train_indices = list(range(0, 6000))
valid_indices = list(range(6000, 7700))
test_indices = list(range(7700, 9800))

model = LES(input_channels = input_length*2, output_channels = 2, kernel_size = kernel_size, 
            dropout_rate = dropout_rate, time_range = time_range).to(device)
model = nn.DataParallel(model, device_ids=[0])

from s3fs.core import S3FileSystem
s3 = S3FileSystem(
    key='jMc2Bgylpg3eyeAHV5Cu',                                                                       
    secret='V3qP2YcCkpK6SJp7LOZlxdBTaQ2tR5i74xNEjDij',                                                
    client_kwargs={                                                                                   
        'endpoint_url': 'https://rosedata.ucsd.edu',                                                  
        'region_name': 'US'                                                                           
    }
)

def data_gen():
    if not exists('TF-net/rbc_data.pt'):
        s3.download('tfnet/rbc_data.pt', 'TF-net/rbc_data.pt')
    data = torch.load('TF-net/rbc_data.pt')

    # standardization
    std = torch.std(data)
    avg = torch.mean(data)
    data = (data - avg) / std
    data = data[:, :, ::4, ::4]

    # divide each rectangular snapshot into 7 subregions
    # data_prep shape: num_subregions * time * channels * w * h
    data_prep = torch.stack([data[:, :, :, k * 64:(k + 1) * 64] for k in range(7)])

    # use sliding windows to generate 9870 samples
    # training 6000, validation 2000, test 1870
    samples = []
    for j in range(0, 1510 - 100):
        for i in range(7):
            samples.append(data_prep[i, j: j + 100])

    return samples


samples = data_gen()
train_set = Dataset(samples, train_indices, input_length + time_range - 1, 40, output_length, True)
valid_set = Dataset(samples, valid_indices, input_length + time_range - 1, 40, 6, True)
train_loader = data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 8)
valid_loader = data.DataLoader(valid_set, batch_size = batch_size, shuffle = False, num_workers = 8)
loss_fun = torch.nn.MSELoss()
regularizer = DivergenceLoss(torch.nn.MSELoss())
coef = 0

optimizer = torch.optim.Adam(model.parameters(), learning_rate, betas = (0.9, 0.999), weight_decay = 4e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.9)

train_mse = []
valid_mse = []
test_mse = []
for i in trange(5, file=sys.stdout):
    start = time.time()
    torch.cuda.empty_cache()
    scheduler.step()
    model.train()
    train_mse.append(train_epoch(train_loader, model, optimizer, loss_fun, coef, regularizer))#
    model.eval()
    mse, preds, trues = eval_epoch(valid_loader, model, loss_fun)
    valid_mse.append(mse)
    if valid_mse[-1] < min_mse:
        min_mse = valid_mse[-1] 
        best_model = model 
        # torch.save(best_model, "model.pth")
    end = time.time()
    if (len(train_mse) > 50 and np.mean(valid_mse[-5:]) >= np.mean(valid_mse[-10:-5])):
            break
    print(train_mse[-1], valid_mse[-1], round((end-start)/60,5))
print(time_range, min_mse)
