from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from copy import deepcopy
from evaluate import evaluate_HIV, evaluate_HIV_population
import os
import argparse

# Initialisation de l'environnement
env = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity  # capacité maximale du buffer
        self.data = []
        self.index = 0  # index de la prochaine cellule à remplir
        self.device = device

    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x: torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))

    def __len__(self):
        return len(self.data)
    
class DQN(nn.Module):
    def __init__(self, env, hidden_size, depth):
        super(DQN, self).__init__()
        self.in_layer = nn.Linear(env.observation_space.shape[0], hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(depth - 1)])
        self.out_layer = nn.Linear(hidden_size, env.action_space.n)

    def forward(self, x):
        x = F.relu(self.in_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        return self.out_layer(x)

class ProjectAgent:
    def __init__(self, config=None):
        # Valeurs par défaut
        default_config = {
            'max_episode': 1000,
            'model_name': 'best_agent',
            'gamma': 0.95,
            'batch_size': 512,
            'buffer_size': 1000000,
            'epsilon_max': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay_period': 1000,
            'epsilon_delay_decay': 20,
            'hidden_size': 256,
            'depth': 5,
            'learning_rate': 0.001,
            'gradient_steps': 1,
            'update_target_strategy': 'replace',
            'update_target_freq': 20,
            'update_target_tau': 0.005,
            'monitoring_nb_trials': 0,
            'monitoring_freq': 10,
            'criterion': torch.nn.SmoothL1Loss(),
            'delay_save': 100,
        }

        # Fusionner config avec les valeurs par défaut
        self.config = {**default_config, **(config or {})}

        # Paramètres de l'agent
        self.max_episode = self.config['max_episode']
        self.model_name = self.config['model_name']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = self.config['gamma']
        self.batch_size = self.config['batch_size']
        self.buffer_size = self.config['buffer_size']
        self.memory = ReplayBuffer(self.buffer_size, self.device)
        self.epsilon_max = self.config['epsilon_max']
        self.epsilon_min = self.config['epsilon_min']
        self.epsilon_stop = self.config['epsilon_decay_period']
        self.epsilon_delay = self.config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop
        self.hidden_size = self.config['hidden_size']
        self.depth = self.config['depth']
        self.model = DQN(env, self.hidden_size, self.depth).to(self.device)
        self.target_model = deepcopy(self.model).to(self.device)
        self.criterion = self.config['criterion']
        self.lr = self.config['learning_rate']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.nb_gradient_steps = self.config['gradient_steps']
        self.update_target_strategy = self.config['update_target_strategy']
        self.update_target_freq = self.config['update_target_freq']
        self.update_target_tau = self.config['update_target_tau']
        self.monitoring_nb_trials = self.config['monitoring_nb_trials']
        self.monitoring_freq = self.config['monitoring_freq']
        self.delay_save = self.config['delay_save']

    def greedy_action(self, network, state):
        with torch.no_grad():
            Q = network(torch.Tensor(state).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item() 
        
    def act(self, observation, use_random=False):
        if use_random:
            return env.action_space.sample()
        else:
            return self.greedy_action(self.model, observation)

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        best_score = 0
        while episode < max_episode:
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            if np.random.rand() < epsilon:
                action = self.act(state, use_random=True)
            else:
                action = self.act(state, use_random=False)
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            if self.update_target_strategy == 'replace' and step % self.update_target_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            elif self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                for key in model_state_dict:
                    target_state_dict[key] = self.update_target_tau * model_state_dict[key] + (1 - self.update_target_tau) * target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            step += 1
            if done or trunc:
                episode += 1
                print(f"Episode {episode}, epsilon {epsilon:.4f}, episode return {episode_cum_reward:.2f}")
                score_agent = evaluate_HIV(agent=self, nb_episode=1)
                if episode > self.delay_save and score_agent > best_score:
                    best_score = score_agent
                    self.save(f"{self.model_name}.pth")
                    print(f"Best score updated: {best_score:.2e}")
                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        self.model.load_state_dict(torch.load(f"{self.model_name}.pth", map_location=self.device))#f"{os.getcwd()}/src/" + self.model_name + '.pth', map_location='cpu')
        self.model.eval()

if __name__ == "_main_":
    config = {
        'model_name': 'best_agent', 
        'max_episode': 1000,
        'hidden_size': 256,
        'depth': 5,
    }

    agent = ProjectAgent(config)
    agent.train(env, config['max_episode'])

