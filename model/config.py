import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import random
import copy
import os
from torchvision import models, transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
from transformers import BertModel, BertTokenizer
warnings.filterwarnings("ignore")

padding_idx = 19587
SOS_token = 19585
EOS_token = 19586
UNK_token = 19584
word_cnt = 19588
batch_size = 8
num_epoch = 200
device = torch.device("cuda")

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
