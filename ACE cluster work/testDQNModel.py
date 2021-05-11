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
import base64
# import io
import gym
# from IPython import display
#from pyvirtualdisplay import Display
# import copy 
# from tqdm import tqdm
import time

class Model(nn.Module):
  #takes the # of frames stacked and the possible outputs (move right, left, etc)
  def __init__(self, numberStacked, possibleOutputs, bSize, initSize):
    super(Model, self).__init__()
    hiddenKernels = 8
    self.bSize = bSize
    self.initSize = initSize

    self.sizePostConvolution = 30752
    self.conv1 = nn.Conv2d(numberStacked, hiddenKernels, 2)
    self.rl = nn.ReLU()
    self.conv2 = nn.Conv2d(hiddenKernels, hiddenKernels, 2)
    self.fc1 = nn.Linear(self.sizePostConvolution, possibleOutputs)

  def forward(self, x):
    x = x.view(-1, 4, self.initSize, self.initSize)
    #print(x.size())
    x = self.conv1(x)
    x = self.rl(x)
    #print(x.size())
    x = self.conv2(x)
    #print(x.size())

    x = x.view(-1, self.sizePostConvolution)
    #print(x.size())
    #x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
    #print(x.size())
    x = self.fc1(x)
    return x


def preprocessing(input_next_state):
    next_state = input_next_state.__array__()

    next_state = torch.tensor(next_state, dtype=torch.float)/255
    next_state = next_state.view(1, 4, 64, 64)
    # r = random.randint(1, 100)
    # if r == 55:
    #     plt.imshow(next_state.numpy()[0][0], interpolation='nearest')
    #     plt.savefig('testPng.png')
    #     quit()
    return next_state

# def execute_action(action, state, test_env):
#     next_state, reward, done, _ = test_env.step(action)
#     if reward != 0:
#         print(reward)
#     next_state = preprocessing(next_state)
#     return next_state, reward, done, test_env

# def action_value(model, input_state):
#     q_val = model(input_state)
#     action = torch.argmax(q_val)
#     return action

games = {'breakout-dqnWeights.pt': [4, 4, 32, 64, 'Breakout-v0'], 'Tut-dqnWeights.pt': [4, 8, 32, 64, 'Tutankham-v0'], 'tank-dqnWeights.pt': [4, 18, 32, 64, 'Robotank-v0']}
thisGame = 'tank-dqnWeights.pt'
inputs = games[thisGame]
loadedData = torch.load(thisGame)
m = Model(inputs[0],inputs[1],inputs[2],inputs[3])
m.load_state_dict(loadedData['model_state_dict'])
print(f'Scores achived when training were: {loadedData["scoreEachEp"]}')
GameName = inputs[4]
test_env = gym.make(GameName)
test_env = ResizeObservation(test_env, 64)
test_env = GrayScaleObservation(test_env)
test_env = FrameStack(test_env, 4) #stack 4 most recent frames
tries = 2
time_in_episode = 1000000
scores = []
m.eval()
with torch.no_grad():
    for _ in range(tries):
        print('ep start')
        lastState = None
        score = 0
        time_step = 0
        episodeDone = False
        next_state = test_env.reset()   
        while not episodeDone:
            next_state = preprocessing(next_state)
            if time_step > 50:
                quit()
            # quit()
            # if lastState != None:
            #     if (lastState.numpy() == next_state.numpy()).all():
            #         print('last state = next state')
            #         plt.imshow(next_state.numpy()[0][0], interpolation='nearest')
            #         plt.savefig('testPng.png')
            #         quit()
            lastState = next_state
            time_step += 1 
            q = m(next_state)
            # print(q)
            action = torch.argmax(q)
            print(action)
            next_state, reward, done, _ = test_env.step(action)
            if reward != 0:
                print(f'reward: {reward}')
            # action = test_env.action_space.sample()
            # print(action)
            # next_state, reward, done, _ = test_env.step(action)
            score += reward
            if time_step >= time_in_episode or done:
                print(f'finished at tStep {time_step} and done = {done}')
                episodeDone = True
        scores.append(score)
print(scores)
print(sum(scores)/len(scores))