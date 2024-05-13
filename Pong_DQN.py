import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(self._get_conv_output((1, input_channels, 80, 80)), 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
    def _get_conv_output(self, shape):
        input = torch.rand(shape)
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        return int(np.prod(output.size()))

# ゲーム環境の初期化
env = gym.make('PongNoFrameskip-v4')

# ネットワーク入力と出力の次元を定義
input_channels = 1  # Pongゲームの画像チャンネル数
output_dim = env.action_space.n  # 取りうるアクションの数

# ネットワークとオプティマイザの初期化
policy_net = DQN(input_channels, output_dim)
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

# これでエラーが解消されるはずです。ネットワークのフラット化された特徴サイズが正確に計算されます。


def preprocess_frame(frame):
    # 画像の形状とタイプを確認
    if frame.ndim == 3 and frame.shape[2] == 3:
        frame = frame[35:195]
        frame = frame[::2, ::2, 0]
        frame[frame == 144] = 0
        frame[frame == 109] = 0
        frame[frame != 0] = 1
        return frame.astype(np.float32).reshape(1, 80, 80)
    else:
        raise ValueError("Unexpected frame format: {}", format(frame.shape))

env = gym.make('PongNoFrameskip-v4')
input_channels = 1
output_dim = env.action_space.n

policy_net = DQN(input_channels, output_dim)
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
epsilon = 0.1

memory = deque(maxlen=10000)

def optimize_model():
    if len(memory) < 100:
        return
    transitions = random.sample(memory, 32)
    batch = list(zip(*transitions))
    state_batch = torch.tensor(np.array(batch[0]), dtype=torch.float32)
    action_batch = torch.tensor(batch[1])
    reward_batch = torch.tensor(batch[2], dtype=torch.float32)
    next_state_batch = torch.tensor(np.array(batch[3]), dtype=torch.float32)

    state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
    next_state_values = policy_net(next_state_batch).max(1)[0].detach()
    expected_state_action_values = (next_state_values * 0.99) + reward_batch

    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

for episode in range(200):
    raw_state = env.reset()[0]  # 画像データの取り出し
    state = preprocess_frame(raw_state)
    total_reward = 0
    done = False
    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = policy_net(torch.from_numpy(state).unsqueeze(0)).argmax().item()

        next_raw_state, reward, done, _, info = env.step(action)  # 返されるタプルの長さに対応するために変更
        next_state = preprocess_frame(next_raw_state)
        memory.append((state, action, reward, next_state))
        state = next_state
        total_reward += reward
        optimize_model()

    print(f'Episode: {episode}, Total Reward: {total_reward}')
    epsilon *= 0.99

env.close()
