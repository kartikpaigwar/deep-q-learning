#lr change to 1e-4
#target update is added
import gym
import math
import random
from collections import deque
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
import sys
import time

frame_skip = int(sys.argv[1])
env = gym.make('BreakoutNoFrameskip-v4')

# set up matplotlib
# is_ipython = 'inline' in matplotlib.get_backend()

# if is_ipython:
#     from IPython import display

# plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)




class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.n_action = 4

        self.conv1 = nn.Conv2d(3, 16, kernel_size=8, stride=4, padding=0)  # (In Channel, Out Channel, ...)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=0)

        self.affine1 = nn.Linear(1152, 256)
        self.affine2 = nn.Linear(256, self.n_action)

        nn.init.xavier_uniform_(self.conv1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.conv2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.conv3.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.affine1.weight,
                                gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self.affine2.weight,
                                gain=nn.init.calculate_gain('linear'))



    def forward(self, x):
        # print(x.shape)
        h = F.relu(self.conv1(x))
        # print(h.shape)
        h = F.relu(self.conv2(h))
        # print(h.shape)
        h = F.relu(self.conv3(h))
        # print(h.shape)

        # print(h.size())
        # print(h.view(h.size(0), -1).size())

        h = F.relu(self.affine1(h.view(h.size(0), -1)))
        # print(h.shape)
        h = self.affine2(h)

        return h


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


def get_screen(screen):
    # screen = env.render(mode='rgb_array')
    # env.render()
    # time.sleep(1/30)
    # screen = screen.transpose((2, 0, 1))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    # screen = screen[0]
    # Strip off the top of the screen
    screen_height, screen_width = screen.shape
    screen = screen[int(screen_height*0.2):int(screen_height)]
    screen = cv2.resize(screen, (0, 0), fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    return screen

s = deque(maxlen = 3)

def stackframes():
    for _ in range(3):
        init_frame = env.reset()
        next_frame = get_screen(init_frame)
        s.append(next_frame)
    buff = np.array(s)
    stack = torch.from_numpy(buff)
    return stack.unsqueeze(0)




######################################################################################
# env.reset()                                                                        #
# s1 = get_screen()      # 134 * 128                                                 #
# cv2.imshow("extracted",s1)
# cv2.waitKey(0)
# exit()
######################################################################################

#Parameters

BATCH_SIZE = 36
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TARGET_UPDATE = 1000




# init_screen = stackframes()
# _, _, screen_height, screen_width = init_screen.shape

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr = 5e-4 , weight_decay= 1e-5)
memory = ReplayMemory(60000)


steps_done = 0


def select_action(state):
    global steps_done
    state = state.to(device)
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold and steps_done > 1000:
        with torch.no_grad():
            # t.max(1) will return largest value for column of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(4)]],device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        a = means.numpy()
        if a[-1]>10:
            print(a[-1])
            torch.save(policy_net.state_dict(), './policy_net_modelbreakout.pth')
            torch.save(target_net.state_dict(), './target_net_modelbreakout.pth')
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    # plt.pause(0.001)  # pause a bit so that plots are updated
    plt.savefig('fixeddqnscores.png')

    # if is_ipython:
    #     display.clear_output(wait=True)
    #     display.display(plt.gcf())




def optimize_model():
    if len(memory) < BATCH_SIZE:
        return None
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None]).to(device)
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    reward_batch.data.clamp_(-1, 1)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.data.cpu().numpy()





num_episodes = 15000
for i_episode in range(num_episodes):
    # Initialize the environment and state

    obs = env.reset()
    # env.step(1)    # fire ball at the begining of episode


    epi_reward = 0

    state = stackframes()
    done = False
    epiloss = []
    t = 0
    if memory.__len__() >= 49999:
        print("max buffer size reached")

    while True:
        # Select and perform an action
        action = select_action(state)

        actionvalue = action.item()

        for skip in range(frame_skip):
            obs, reward, done, info = env.step(actionvalue)
            epi_reward = epi_reward + reward

            next_frame = get_screen(obs)
            t = t + 1
            if done:
                break

        s.append(next_frame)

        buff = np.array(s)
        nextstate_stack = torch.from_numpy(buff)
        nextstate_stack = nextstate_stack.unsqueeze(0)

        reward = torch.tensor([reward])

        # Observe new state
        if not done:
            next_state = nextstate_stack
        else:
            next_state = None

        # Store the transition in memory

        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        step_loss = optimize_model()
        if step_loss!= None:
            epiloss.append(step_loss)

        if t>TARGET_UPDATE and t % TARGET_UPDATE <= 3:
            print("Target_Net Updated.....")
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            episode_durations.append(epi_reward)
            print(epi_reward)
            # plot_durations()
            break

    mean_loss = np.mean(epiloss)
    if i_episode % 10 == 0:
        print("Episode "+ str(i_episode) + " loss = "+ str(mean_loss))


print('Complete')
# env.render()
env.close()
plt.ioff()
# plt.show()
