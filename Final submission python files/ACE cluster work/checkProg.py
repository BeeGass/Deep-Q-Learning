import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as T
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.wrappers import FrameStack, Monitor, GrayScaleObservation, ResizeObservation
import random
import atari_py
import os
import base64
directory_path = os.getcwd()
print(directory_path)
names = ['breakout', 'tank',  'King tut']#'tennis',
fPath = ['breakout-dqnWeights.pt', 'tank-dqnWeights.pt',  'Tut-dqnWeights.pt']#'TennisdqnWeights.pt',
for n, p in zip(names, fPath):
    print(f'{n} has scores:')
    x = torch.load(p)
    print(x['scoreEachEp'])