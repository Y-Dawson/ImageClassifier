import pandas as pd
import torch
import os
import csv

from visualizer import get_local
get_local.activate()
import numpy as np
from helpers import *
from layers import *
from constants import *
from ViT import *
from trace_utils import *
from ResMLP import *
from ResNet import *
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import gradcam
ViT = VisionTransformer(representation_size = None, patch_size= 16, embed_dim= 768, depth = 12, num_heads= 12)
#_load_weights(ViT, './ViT.npz')
#print(len(ViT))
ViT.load_state_dict(torch.load('./ViT.pth'))
#print(ViT)

config = resolve_data_config({}, model=ViT)
print('config:', config)
transform = create_transform(**config)
img = Image.open("./Sample.jpg")
tensor = transform(img).unsqueeze(0)
model = ViT
model.eval()
with torch.no_grad():
    out = model(tensor)

cache = get_local.cache
print(list(cache.keys()))
attention_maps = cache['Block.forward']
print('len(attention_maps):', len(attention_maps))
print(attention_maps[0].shape)