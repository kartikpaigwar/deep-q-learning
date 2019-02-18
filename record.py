from collections import deque, namedtuple
import gym
import gym.utils.play as gp
import torch
import cv2
import numpy as np


state = []

histlen = 4

stateHistory = deque(maxlen=histlen)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

maxbuffer = 1000000

counter = 0


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    # def ToTensors(self,state,action,next_state,reward):
    #     tensor_list = [state,action,next_state,reward]
    #     tensor_tensor = torch.stack(tensor_list)
    #     return tensor_tensor
    ''' Let's see what we can do '''

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def __len__(self):
        return len(self.memory)

memory = ReplayMemory(maxbuffer)

def get_screen(screen):
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    screen_height, screen_width = screen.shape
    screen = screen[int(screen_height * 0.2):int(screen_height)]
    screen = cv2.resize(screen, (0, 0), fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    return screen



def Stacking(screen):
    next_frame = get_screen(screen)
    stateHistory.append(next_frame)
    buff = np.array(stateHistory)
    stack = torch.from_numpy(buff)
    return stack.unsqueeze(0)


def callback(prev_obs, obs, action, rew, env_done, info):
    if not env_done:
        current_lives = 0
        for k, v in info.items():
            current_lives = v
        started_flag = 0
        if not (current_lives == 5 and action == 0):
            started_flag = 1
        if started_flag == 1:

            # data_tuple = (prev_obs,action,rew)

            global counter
            global state
            global memory
            if counter < 4:
                state = Stacking(prev_obs)
                counter += 1
            else:
                nextstate = Stacking(obs)
                action = torch.tensor([[action]],dtype=torch.long)
                rew = torch.tensor([rew])
                memory.push(state,action,nextstate,rew)
                state = nextstate


    else:
        print("Environment done once")
        print(type(memory.memory))
        print(type(memory.memory[1]))
        # memory_buffer = np.asarray(memory.memory[1])
        # print(memory_buffer.shape)
        exit()



gp.play(gym.make('BreakoutNoFrameskip-v4'), fps=60, zoom=3, callback=callback)

# fps = 10 now ... Might have to change?
