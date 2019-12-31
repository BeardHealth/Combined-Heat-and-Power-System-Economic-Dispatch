import argparse
from collections import namedtuple
from itertools import count
from test_system4 import Env
import os, sys, random
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='test', type=str) # mode = 'train' or 'test'
parser.add_argument("--env_name", default="test system4")
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--iteration', default=5, type=int)

parser.add_argument('--learning_rate', default=5e-5, type=float)
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--capacity', default=5000, type=int) # replay buffer size
parser.add_argument('--num_iteration', default=3000, type=int) #  num of  games
parser.add_argument('--batch_size', default=64, type=int) # mini batch size
parser.add_argument('--seed', default=42, type=int)

# optional parameters
parser.add_argument('--num_hidden_layers', default=2, type=int)
parser.add_argument('--sample_frequency', default=256, type=int)
parser.add_argument('--activation', default='Relu', type=str)
parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--load', default=False, type=bool) # load model
parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
parser.add_argument('--policy_noise', default=0.2, type=float)
parser.add_argument('--noise_clip', default=0.5, type=float)
parser.add_argument('--policy_delay', default=2, type=int)
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_episode', default=500, type=int)
parser.add_argument('--print_log', default=5, type=int)
args = parser.parse_args()

# Set seeds
# env.seed(args.seed)
# torch.manual_seed(args.seed)
# np.random.seed(args.seed)'
script_name = os.path.basename(__file__)
env = Env()
env.construct()
num_state_h = env.state_dim_h
num_state_m = env.state_dim_m
num_state_p = env.state_dim_p
num_action = env.action_dim
max_action = 0.5
min_Val = torch.tensor(1e-7).float().to(device)  # min value

directory = './exp' + script_name + args.env_name +'./'
'''
Implementation of TD3 with pytorch 
Original paper: https://arxiv.org/abs/1802.09477
'''


class Replay_buffer():
    """
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    """
    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        state_p, state_m, state_h, next_state_p, next_state_m, next_state_h, action, reward, done = [], [], [], [], [], [], [], [], []

        for i in ind:
            p, m, h, n_p, n_m, n_h, a, r, d = self.storage[i]
            state_p.append(np.array(p, copy=False))
            state_m.append(np.array(m, copy=False))
            state_h.append(np.array(h, copy=False))
            next_state_p.append(np.array(n_p, copy=False))
            next_state_m.append(np.array(n_m, copy=False))
            next_state_h.append(np.array(n_h, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            done.append(np.array(d, copy=False))

        return np.array(state_p), np.array(state_m), np.array(state_h), np.array(next_state_p), np.array(next_state_m), np.array(next_state_h), \
               np.array(action), np.array(reward).reshape(-1, 1), np.array(done).reshape(-1, 1)


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        # state encoder
        self.h1 = nn.Linear(num_state_h, 128)
        self.h2 = nn.Linear(num_state_h, 128)
        self.h3 = nn.Linear(num_state_h, 128)
        self.m1 = nn.Linear(num_state_m, 128)
        self.m2 = nn.Linear(num_state_m, 128)
        self.m3 = nn.Linear(num_state_m, 128)
        self.m4 = nn.Linear(num_state_m, 128)
        self.m4 = nn.Linear(num_state_m, 128)
        self.p1 = nn.Linear(num_state_p, 128)
        self.p2 = nn.Linear(num_state_p, 128)
        self.p3 = nn.Linear(num_state_p, 128)
        self.p4 = nn.Linear(num_state_p, 128)
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # policy head
        self.policy_conv1 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=1)
        self.policy_bn1 = nn.BatchNorm1d(16)
        self.action = nn.Linear(16*11, num_action)

    def forward(self, states_p, states_m, states_h):
        p1 = F.relu(self.p1(states_p[:, 0, :])).unsqueeze(1)
        p2 = F.relu(self.p2(states_p[:, 1, :])).unsqueeze(1)
        p3 = F.relu(self.p3(states_p[:, 2, :])).unsqueeze(1)
        p4 = F.relu(self.p4(states_p[:, 3, :])).unsqueeze(1)
        h1 = F.relu(self.h1(states_h[:, 0, :])).unsqueeze(1)
        h2 = F.relu(self.h2(states_h[:, 1, :])).unsqueeze(1)
        h3 = F.relu(self.h3(states_h[:, 2, :])).unsqueeze(1)
        m1 = F.relu(self.m1(states_m[:, 0, :])).unsqueeze(1)
        m2 = F.relu(self.m2(states_m[:, 1, :])).unsqueeze(1)
        m3 = F.relu(self.m3(states_m[:, 2, :])).unsqueeze(1)
        m4 = F.relu(self.m4(states_m[:, 3, :])).unsqueeze(1)
        input_cnn = torch.cat((p1, p2, p3, p4, h1, h2, h3, m1, m2, m3, m4), 1).permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(input_cnn)))
        # policy head
        p = self.relu(self.policy_bn1(self.policy_conv1(x)))
        p = p.view(p.shape[0], 16*11)
        action = torch.tanh(self.action(p))*max_action

        return action


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        # state encoder
        self.h1 = nn.Linear(num_state_h, 128)
        self.h2 = nn.Linear(num_state_h, 128)
        self.h3 = nn.Linear(num_state_h, 128)
        self.m1 = nn.Linear(num_state_m, 128)
        self.m2 = nn.Linear(num_state_m, 128)
        self.m3 = nn.Linear(num_state_m, 128)
        self.m4 = nn.Linear(num_state_m, 128)
        self.m4 = nn.Linear(num_state_m, 128)
        self.p1 = nn.Linear(num_state_p, 128)
        self.p2 = nn.Linear(num_state_p, 128)
        self.p3 = nn.Linear(num_state_p, 128)
        self.p4 = nn.Linear(num_state_p, 128)
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # value head
        self.value_conv1 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=1)
        self.value_bn1 = nn.BatchNorm1d(16)
        self.value_fc1 = nn.Linear(16*11, 32)
        self.value_fc2 = nn.Linear(32 + num_action, 1)

    def forward(self, states_p, states_m, states_h, action):
        p1 = F.relu(self.p1(states_p[:, 0, :])).unsqueeze(1)
        p2 = F.relu(self.p2(states_p[:, 1, :])).unsqueeze(1)
        p3 = F.relu(self.p3(states_p[:, 2, :])).unsqueeze(1)
        p4 = F.relu(self.p4(states_p[:, 3, :])).unsqueeze(1)
        h1 = F.relu(self.h1(states_h[:, 0, :])).unsqueeze(1)
        h2 = F.relu(self.h2(states_h[:, 1, :])).unsqueeze(1)
        h3 = F.relu(self.h3(states_h[:, 2, :])).unsqueeze(1)
        m1 = F.relu(self.m1(states_m[:, 0, :])).unsqueeze(1)
        m2 = F.relu(self.m2(states_m[:, 1, :])).unsqueeze(1)
        m3 = F.relu(self.m3(states_m[:, 2, :])).unsqueeze(1)
        m4 = F.relu(self.m4(states_m[:, 3, :])).unsqueeze(1)
        input_cnn = torch.cat((p1, p2, p3, p4, h1, h2, h3, m1, m2, m3, m4), 1).permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(input_cnn)))
        # value head
        v = self.relu(self.value_bn1(self.value_conv1(x)))
        v = v.view(v.shape[0], 16 * 11)
        v = self.relu(self.value_fc1(v))
        state_action = torch.cat([v, action], 1)
        value = self.value_fc2(state_action)

        return value


class TD3(object):
    def __init__(self):

        self.actor = Actor().to(device)
        self.actor_target = Actor().to(device)
        self.critic_1 = Critic().to(device)
        self.critic_1_target = Critic().to(device)
        self.critic_2 = Critic().to(device)
        self.critic_2_target = Critic().to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.learning_rate)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=args.learning_rate)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=args.learning_rate)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.max_action = max_action
        self.memory = Replay_buffer(args.capacity)
        self.writer = SummaryWriter(directory)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state_p, state_m, state_h):
        state_p = torch.from_numpy(state_p).float().unsqueeze(0).to(device)
        state_m = torch.from_numpy(state_m).float().unsqueeze(0).to(device)
        state_h = torch.from_numpy(state_h).float().unsqueeze(0).to(device)

        return self.actor(state_p, state_m, state_h).cpu().data.numpy().flatten()

    def update(self, num_iteration):

        if self.num_training % 500 == 0:
            print("====================================")
            print("model has been trained for {} times...".format(self.num_training))
            print("====================================")
        for i in range(num_iteration):
            p, m, h, n_p, n_m, n_h, a, r, d = self.memory.sample(args.batch_size)
            state_p = torch.FloatTensor(p).to(device)
            state_m = torch.FloatTensor(m).to(device)
            state_h = torch.FloatTensor(h).to(device)
            action = torch.FloatTensor(a).to(device)
            next_state_p = torch.FloatTensor(n_p).to(device)
            next_state_m = torch.FloatTensor(n_m).to(device)
            next_state_h = torch.FloatTensor(n_h).to(device)
            done = torch.FloatTensor(d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Select next action according to target policy:
            noise = torch.ones_like(action).data.normal_(0, args.policy_noise).to(device)
            noise = noise.clamp(-args.noise_clip, args.noise_clip)
            next_action = (self.actor_target(next_state_p, next_state_m, next_state_h) + noise)

            next_action = next_action.clamp(-max_action, max_action)

            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state_p, next_state_m, next_state_h, next_action)
            target_Q2 = self.critic_2_target(next_state_p, next_state_m, next_state_h, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * args.gamma * target_Q).detach()

            # Optimize Critic 1:
            current_Q1 = self.critic_1(state_p, state_m, state_h, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q).to(device)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            self.writer.add_scalar('Loss/Q1_loss', loss_Q1, global_step=self.num_critic_update_iteration)

            # Optimize Critic 2:
            current_Q2 = self.critic_2(state_p, state_m, state_h, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q).to(device)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            self.writer.add_scalar('Loss/Q2_loss', loss_Q2, global_step=self.num_critic_update_iteration)
            # Delayed policy updates:
            if i % args.policy_delay == 0:
                # Compute actor loss:
                actor_loss = - self.critic_1(state_p, state_m, state_h, self.actor(state_p, state_m, state_h)).mean().to(device)

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(((1 - args.tau) * target_param.data) + args.tau * param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(((1 - args.tau) * target_param.data) + args.tau * param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(((1 - args.tau) * target_param.data) + args.tau * param.data)

                self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1
        self.num_training += 1

    def save(self):
        torch.save(self.actor.state_dict(), directory+'actor.pth')
        torch.save(self.actor_target.state_dict(), directory+'actor_target.pth')
        torch.save(self.critic_1.state_dict(), directory+'critic_1.pth')
        torch.save(self.critic_1_target.state_dict(), directory+'critic_1_target.pth')
        torch.save(self.critic_2.state_dict(), directory+'critic_2.pth')
        torch.save(self.critic_2_target.state_dict(), directory+'critic_2_target.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.actor_target.load_state_dict(torch.load(directory + 'actor_target.pth'))
        self.critic_1.load_state_dict(torch.load(directory + 'critic_1.pth'))
        self.critic_1_target.load_state_dict(torch.load(directory + 'critic_1_target.pth'))
        self.critic_2.load_state_dict(torch.load(directory + 'critic_2.pth'))
        self.critic_2_target.load_state_dict(torch.load(directory + 'critic_2_target.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


def main():
    # initial
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    agent = TD3()
    ep_r = 0

    if args.mode == 'test':
        agent.load()
        for i in range(1):
            state = env.reset()
            for t in count():
                action = agent.select_action(state[0], state[1], state[2])
                action = np.float32(action)
                next_state, reward, done = env.step(np.float32(action))
                ep_r += reward
                if done or t == 500:
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    print(env._get_obs())
                    break
                state = next_state
        for i in range(4):
            print(env.power_only[i].get())
            print(env.CHP[i].get())
        for i in range(3):
            print(env.heat_only[i].get())


    elif args.mode == 'train':
        best_value = -10000000
        print("====================================")
        print("Collection Experience...")
        print("====================================")
        if args.load: agent.load()
        for i in range(args.num_iteration):
            state = env.reset()
            for t in range(500):
                action = agent.select_action(state[0], state[1], state[2])
                action = action + np.random.normal(args.exploration_noise, args.exploration_noise, size=num_action)
                action = action.clip(-max_action, max_action)
                next_state, reward, done = env.step(action)
                ep_r += reward
                agent.memory.push((state[0], state[1], state[2], next_state[0], next_state[1], next_state[2], action, reward, np.float(done)))
                if i+1 % 10 == 0:
                    print('Episode {},  The memory size is {} '.format(i, len(agent.memory.storage)))
                state = next_state
                if done or t == args.max_episode-1:
                    if len(agent.memory.storage) >= args.capacity - 1:
                        agent.update(10)
                    agent.writer.add_scalar('ep_r', ep_r, global_step=i)
                    if i % args.print_log == 0:
                        print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    mean_reward = ep_r
                    ep_r = 0
                    break

            if mean_reward > best_value and done:
                best_value = mean_reward
                print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, mean_reward, t))
                agent.save()

    else:
        raise NameError("mode wrong!!!")


if __name__ == '__main__':
    main()