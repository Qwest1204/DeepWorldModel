from torch.distributions import Normal

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

class PPO(nn.Module):
    def __init__(self, policy_net, value_net, state_dim, action_dim, gamma=0.99, batch_size=100, epsilone=0.2):
        super(PPO, self).__init__()
        self.policy_net = policy_net
        self.value_net = value_net
        self.transform = transforms.Compose([
                                             transforms.ToPILImage(),
                                             transforms.Resize(96),
                                             transforms.ToTensor(),
                                             ])

        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilone = epsilone

        self.policy_opt = torch.optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.value_opt = torch.optim.Adam(self.value_net.parameters(), lr=3e-4)

    def get_action(self, state):
        state = self.transform(state)
        print(state)
        with torch.no_grad():
            mean, log_std = self.policy_net(state.unsqueeze(0))
        dist = Normal(mean, torch.exp(log_std))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.numpy(), log_prob.numpy().squeeze()

    def compute_returns(self, rewards, dones):
        returns = np.zeros_like(rewards)
        running_return = 0

        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * (1 - dones[t])
            returns[t] = running_return

        return returns

    def fit(self, states, actions, rewards, dones, old_log_probs):
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        old_log_probs = np.array(old_log_probs)

        returns = self.compute_returns(rewards, dones)

        states = torch.stack([self.transform(img) for img in states])

        actions_tensor = torch.FloatTensor(actions)
        returns_tensor = torch.FloatTensor(returns).unsqueeze(1)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).unsqueeze(1)

        n_samples = len(states)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        all_values = self.value_net(states)

        advantages = returns_tensor - all_values.detach()

        for start in range(0, n_samples, self.batch_size):
            end = start + self.batch_size
            batch_indices = indices[start:end]

            # Батч данных
            b_states = states[batch_indices]
            b_actions = actions_tensor[batch_indices]
            b_returns = returns_tensor[batch_indices]
            b_old_log_probs = old_log_probs_tensor[batch_indices]
            b_advantage = advantages[batch_indices]

            b_mean, b_log_std = self.policy_net(b_states)
            b_dist = Normal(b_mean, torch.exp(b_log_std))
            b_new_log_probs = b_dist.log_prob(b_actions)

            if len(b_new_log_probs.shape) > 1:
                b_new_log_probs = b_new_log_probs.sum(dim=-1, keepdim=True)

            b_ratio = torch.exp(b_new_log_probs - b_old_log_probs)

            policy_loss_1 = b_ratio * b_advantage
            policy_loss_2 = torch.clamp(b_ratio, 1. - self.epsilone, 1. + self.epsilone) * b_advantage
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()


            b_values = self.value_net(b_states)
            value_loss = torch.nn.functional.mse_loss(b_values, b_returns)

            self.value_opt.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)  # Clip gradients
            self.value_opt.step()

