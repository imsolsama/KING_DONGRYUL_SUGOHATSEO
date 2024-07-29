import math
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# GPU를 사용할 경우
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transition을 저장하기 위한 namedtuple입니다! 아래 ReplayMemory 클래스에서 사용해요. 
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# ReplayMemory 클래스 구현
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Transition 저장"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """무작위로 배치 샘플링"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """저장된 경험의 수 반환"""
        return len(self.memory)

# DQN 모델 구현
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_observations, 128)  # 입력층
        self.fc2 = nn.Linear(128, 128)              # 은닉층
        self.fc3 = nn.Linear(128, n_actions)        # 출력층

    def forward(self, x):
        x = F.relu(self.fc1(x))  # ReLU 활성화 함수
        x = F.relu(self.fc2(x))  # ReLU 활성화 함수
        return self.fc3(x)       # 출력층

class DQNAgent:
    def __init__(self, state_size, action_size, eps_start, eps_end, eps_decay, gamma, lr, batch_size, tau):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(10000)
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.steps_done = 0
        self.episode_rewards = []
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.update_target_net()

        # 플로팅 초기화
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.reward_line, = self.ax.plot([], [], label='Total Reward')
        self.ax.legend()
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Total Reward')

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def plot_rewards(self):
        self.reward_line.set_xdata(range(len(self.episode_rewards)))
        self.reward_line.set_ydata(self.episode_rewards)

        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
