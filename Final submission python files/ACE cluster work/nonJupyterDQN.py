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
    #layer_one_hiddenKernels = 54
    #layer_two_hiddenKernels = 6

    #sizePostConvolution = 525824 
    #sizePostConvolution = 262912
    self.sizePostConvolution = 30752
    #sizePostConvolution = 134400
    #self.conv1 = nn.Conv2d(in_channels = numberStacked, out_channels = layer_one_hiddenKernels, kernel_size = 2, stride = 2, padding = 1)
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

class Agent():
  def __init__(self, size, numberStacked, possibleOutputs, gamma, initSize):
    self.replay_buffer_size = size
    self.replay_buffer_list = []
    self.batch_size = 32
    self.initSize = initSize
    self.model = Model(numberStacked, possibleOutputs, self.batch_size, self.initSize)
    self.targetModel = Model(numberStacked, possibleOutputs, self.batch_size, self.initSize)
    self.targetModel.load_state_dict(self.model.state_dict())
    self.targetModel.eval()
    self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
    self.lossFn = torch.nn.MSELoss()
    self.gamma = gamma
    self.loss = 0

  def action_value(self, input_state):
    with torch.no_grad():
        #print(input_state.size())
        q_val = self.model(input_state)
    #print("q_val: ", q_val)
    action = torch.argmax(q_val)
    #print("action: ", action)
    return action

  def sample_replay_buffer(self, batch_size):
    mini_batch = random.sample(self.replay_buffer_list, batch_size)
    return mini_batch

  def SGD(self):
    mini_batch = self.sample_replay_buffer(self.batch_size)
    for batch in mini_batch:
        self.optimizer.zero_grad()
        state, action, reward, next_state, done = batch
        predicted_q_val = self.model(state)
        predicted_reward = predicted_q_val[0][torch.argmax(predicted_q_val)]
        yj = torch.FloatTensor([reward])
#         print("the reward: ", reward)
        if done:
#             if isinstance(yj, float):
#                 print("this is yj", yj)
            loss = self.lossFn(predicted_reward, yj.detach())
        else:
            target_q_val = self.model(next_state)
            #print("target_q_val: ", target_q_val)
            yj += self.gamma * torch.argmax(target_q_val)
            loss = self.lossFn(predicted_reward, yj.detach())
        loss.backward()
        self.optimizer.step()
    return loss
  
  def addToReplay(self, newInput):
    if len(self.replay_buffer_list) >= self.replay_buffer_size:#random eviction
        #toEvict = random.randint(0, self.replay_buffer_size - 1)
        del self.replay_buffer_list[0]
    self.replay_buffer_list.append(newInput)

  def updateTarget(self):
    self.targetModel.load_state_dict(self.model.state_dict())

  def loadOldValues(self, rb, modelValues):
    self.replay_buffer_list = rb
    self.model.load_state_dict(modelValues)
    self.updateTarget()

class DQN():
    def __init__(self):
        self.episodes = 5000
        self.startEp = 0
        self.time_in_episode = 1000000
        self.epsilon = 1.0
        self.possibleOutputs = 6
        self.gamma = 0.9
        self.rbSize = 150000
        self.numberStacked = 4
        # self.height = 210
        # self.width = 160      
        self.height = 64
        self.width = 64
        self.p_init = self.epsilon
        self.p_end = 0.1
        #self.height = 84
        #self.width = 84
        self.agent = Agent(self.rbSize, self.numberStacked, self.possibleOutputs, self.gamma, self.height)
        PONG = 'PongNoFrameskip-v4'
        CARTPOLE = 'CartPole-v1'
        self.scoreEachEp = []

        self.test_env = gym.make(PONG)
        # self.test_env = Monitor(gym.make(PONG),'./', force=True)
        self.test_env = ResizeObservation(self.test_env, 64)
        self.test_env = GrayScaleObservation(self.test_env)
        self.test_env = FrameStack(self.test_env, self.numberStacked) #stack 4 most recent frames
        self.transform = T.Compose([T.Resize((64,64)), T.ToTensor()])#T.Grayscale(), 
        loadedData = torch.load('dqnWeights.pt')
        self.startEp = len(loadedData['scoreEachEp'])
        print(f'Score each ep as of so far: {loadedData["scoreEachEp"]}')
        self.epsilon = loadedData['epsilon']
        self.agent.loadOldValues(loadedData['replay_buffer'], loadedData['model_state_dict'])
        self.scoreEachEp = loadedData["scoreEachEp"]

    def initTransition(self):
        state = self.test_env.reset()   
        action = self.test_env.action_space.sample()
        next_state, reward, done, _ = self.test_env.step(action)
        grey_scaled_state = self.preprocessing(state)
        grey_scaled_next_state = self.preprocessing(next_state)
        transition = (grey_scaled_state, action, reward, grey_scaled_next_state)
        self.agent.addToReplay((grey_scaled_state, action, reward, grey_scaled_next_state, done))

        return transition

    def execute_action(self, input_action, state):
        next_state, reward, done, _ = self.test_env.step(input_action)
        next_state = self.preprocessing(next_state)
        
        transition = (state, input_action, reward, next_state, done)
        self.agent.addToReplay((state, input_action, reward, next_state, done))

        return transition

    def preprocessing(self, input_next_state):
        next_state = input_next_state.__array__()
        # print(type(next_state))
        # print(next_state.shape)
        # if torch.is_tensor(input_next_state) or not isinstance(input_next_state, gym.wrappers.frame_stack.LazyFrames):
        #     input_next_state = input_next_state.numpy()
        #     next_state = np.transpose(input_next_state, (0, 3, 1, 2))
        # else:
    #        next_state = np.transpose(input_next_state, (0, 3, 1, 2))#batch h w color to batch color h w

#         copy_next_state = np_next_state.copy()

        next_state = torch.tensor(next_state, dtype=torch.float)/255
        next_state = next_state.view(1, self.numberStacked, self.height, self.width)
        # quit()
        return next_state


    def save_weights(self, replay_buffer, epsilon, weights, loss, scoreEachEp):
        the_safe = torch.save({
            'epsilon': epsilon,
            'model_state_dict': weights,
            'replay_buffer': replay_buffer,
            'loss': loss,
            'scoreEachEp': scoreEachEp
            }, 'dqnWeights.pt')
  
    def train(self):
        print('old sgd')
        #next_step = Display()
        #next_step.start()
        for e in range(self.startEp, self.episodes):
            rewardVal = 0
            
            #initialize episode and get first transition
            initial_transition = self.initTransition()
            state, action, reward, next_state = initial_transition

            episodeDone = False
            time_step = 0
#             loss = -100000000000
            totalReward = 0
            startTime = time.time()
            while not episodeDone:
                random_action_prob = random.uniform(0.0, 1.0)
                if random_action_prob < self.epsilon:
                    action = self.test_env.action_space.sample()
                    #print("RANDOM action performed: ", action)
                else: 
                    #perform action for timestep
                    action = self.agent.action_value(next_state)
                    #print("action performed: ", action)
                    
                state, action, reward, next_state, done  = self.execute_action(action, next_state)
                totalReward += reward
                
                #send all information into our replay buffer so we can test on it within SGD
                if time_step % 4 == 0 or reward > 0:
                    self.agent.addToReplay((state, action, reward, next_state, done))
                
                # if ((len(self.agent.replay_buffer) + 1) % self.agent.batch_size) == 0:
                if len(self.agent.replay_buffer_list) >= 20000:
                    loss = self.agent.SGD()
                    
                if time_step >= self.time_in_episode or done:
                    episodeDone = True
                time_step += 1 
            #perform epsilon decay
            episode = e + 1
            epsilon_decay_rate = max((int(2*self.episodes/3) - episode) / int(2*self.episodes/3), 0)
            self.epsilon = (self.p_init - self.p_end) * epsilon_decay_rate + self.p_end
            self.scoreEachEp.append(totalReward)
            print(f'Epoch {e} achieved total score of: {totalReward}, new epsilon is {self.epsilon:.3f}, average score of last 10 eps is: {sum(self.scoreEachEp[-10:]) / len(self.scoreEachEp[-10:])}, episode took {time.time() - startTime} seconds')
            if (e + 1) % 10 == 0:
                self.agent.updateTarget()
                self.save_weights(self.agent.replay_buffer_list, self.epsilon, self.agent.model.state_dict(), loss, self.scoreEachEp)

        #for f in self.test_env.videos:
            #video = io.open(f[0], 'r+b').read()
            #encoded = base64.b64encode(video)

            #display.display(display.HTML(data="""
            #<video alt="test" controls>
            #<source src="data:video/mp4;base64,{0}" type="video/mp4" />
            #</video>
            #""".format(encoded.decode('ascii'))))
            
if __name__ == "__main__":
  torch.device("cuda" if torch.cuda.is_available() else "cpu")
  d = DQN()
  d.train()
