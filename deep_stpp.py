from argparse import ArgumentParser
import os
import sys
import random
import logging
import numpy as np

from scipy import integrate
from sklearn.metrics import mean_squared_error as MSE
import copy

from tqdm.auto import tqdm, trange
from tqdm.contrib import tenumerate

import torch
from torch.utils.data import DataLoader

def imshow(fig):
    return display(SVG(fig.to_image(format="svg")))

sys.path.append("deep-stpp")
from model import DeepSTPP, log_ft, t_intensity, s_intensity
from data.dataset import SlidingWindowWrapper
from data.synthetic import *
from util import *

"""The code below is used to set up customized training device on computer"""
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("You are using GPU acceleration.")
else:
    device = torch.device("cpu")

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
config = Namespace(hid_dim=128, emb_dim=128, out_dim=0, n_layers=1, 
                   lr=0.0003, momentum=0.9, epochs=5, batch=128, opt='Adam', generate_type=True,
                   read_model=False, seq_len=20, eval_epoch=5, s_min=1e-3, b_max=20, 
                   lookahead=1, alpha=0.1, z_dim=128, beta=1e-3, dropout=0, num_head=2,
                   nlayers=3, num_points=20, infer_nstep=10000, infer_limit=13, clip=1.0,
                   constrain_b='sigmoid', sample=True, decoder_n_layer=3)

"""
Prepare logger
"""
logger = logging.getLogger('full_lookahead{}batch{}'.format(config.lookahead, config.batch))
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

# pip install s3fs
from s3fs.core import S3FileSystem
s3 = S3FileSystem(
    key='jMc2Bgylpg3eyeAHV5Cu',                                                                       
    secret='V3qP2YcCkpK6SJp7LOZlxdBTaQ2tR5i74xNEjDij',                                                
    client_kwargs={                                                                                   
        'endpoint_url': 'https://rosedata.ucsd.edu',                                                  
        'region_name': 'US'                                                                           
    } 
)

dataset = 'covid_nj_cases'
key = f'processed/{dataset}.npz'
bucket = 'deep-stpp'
s3.download('{}/{}'.format(bucket, key), 'deep-stpp/covid_nj_cases.npz')

npzf = np.load('deep-stpp/covid_nj_cases.npz', allow_pickle=True)

trainset = SlidingWindowWrapper(npzf['train'], normalized=True)
valset   = SlidingWindowWrapper(npzf['val'],   normalized=True, min=trainset.min, max=trainset.max)
testset  = SlidingWindowWrapper(npzf['test'],  normalized=True, min=trainset.min, max=trainset.max)

train_loader = DataLoader(trainset, batch_size=config.batch, shuffle=True)
val_loader   = DataLoader(valset,   batch_size=config.batch, shuffle=False)
test_loader  = DataLoader(testset,  batch_size=config.batch, shuffle=False)

from model import *
model = DeepSTPP(config, device)

best_model = train(model, train_loader, val_loader, config, logger, device)

