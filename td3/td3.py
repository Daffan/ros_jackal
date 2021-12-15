import copy
import pickle
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_preprocess, head, action_dim):
        super(Actor, self).__init__()

        self.state_preprocess = state_preprocess
        self.head = head
        self.fc = nn.Linear(self.head.feature_dim, action_dim)

    def forward(self, state):
        a = self.state_preprocess(state) if self.state_preprocess else state
        a = self.head(a)
        return torch.tanh(self.fc(a))


class Critic(nn.Module):
    def __init__(self, state_preprocess, head):
        super(Critic, self).__init__()

        # Q1 architecture
        self.state_preprocess1 = state_preprocess
        self.head1 = head
        self.fc1 = nn.Linear(self.head1.feature_dim, 1)

        # Q2 architecture
        self.state_preprocess2 = copy.deepcopy(state_preprocess)
        self.head2 = copy.deepcopy(head)
        self.fc2 = nn.Linear(self.head2.feature_dim, 1)

    def forward(self, state, action):
        state1 = self.state_preprocess1(
            state) if self.state_preprocess1 else state
        sa1 = torch.cat([state1, action], 1)

        state2 = self.state_preprocess2(
            state) if self.state_preprocess2 else state
        sa2 = torch.cat([state2, action], 1)

        q1 = self.head1(sa1)
        q1 = self.fc1(q1)

        q2 = self.head2(sa2)
        q2 = self.fc2(q2)
        return q1, q2

    def Q1(self, state, action):
        state = self.state_preprocess1(
            state) if self.state_preprocess1 else state
        sa = torch.cat([state, action], 1)

        q1 = self.head1(sa)
        q1 = self.fc1(q1)
        return q1


class TD3(object):
    def __init__(
            self,
            actor,
            actor_optim,
            critic,
            critic_optim,
            action_range,
            safe_critic=None,
            safe_critic_optim=None,
            device="cpu",
            gamma=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            n_step=4,
            update_actor_freq=2,
            exploration_noise=0.1,
            safety_threshold=-0.1,
    ):

        self.actor = actor
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = actor_optim

        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = critic_optim

        if safe_critic is not None:
            self.safe_rl = True
            self.safe_critic = safe_critic
            self.safe_critic_target = copy.deepcopy(self.safe_critic)
            self.safe_critic_optim = safe_critic_optim
            self.safety_threshold = safety_threshold

        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.update_actor_freq = update_actor_freq
        self.exploration_noise = exploration_noise
        self.device = device
        self.n_step = n_step

        self.total_it = 0
        self.action_range = action_range
        self._action_scale = torch.tensor(
            (action_range[1] - action_range[0]) / 2.0, device=self.device)
        self._action_bias = torch.tensor(
            (action_range[1] + action_range[0]) / 2.0, device=self.device)

        if self.safe_rl:
            self.grad_dims = [p.numel() for p in self.actor.parameters()]
            n_params = sum(self.grad_dims)
            self.grads = torch.zeros((n_params, 2)).to(self.device)

    def grad2vec(self, grad, i):
        assert self.safe_rl, "[error] Not in Safe RL setting!"
        self.grads[:,i].fill_(0.0)
        beg = 0
        for p, g, dim in zip(self.actor.parameters(), grad, self.grad_dims):
            en = beg + dim
            if g is not None:
                self.grads[beg:en,i].copy_(grad.view(-1).data.clone())
            beg = en

    def vec2grad(self, grad):
        beg = 0
        for p, dim in zip(self.actor.parameters(), self.grad_dims):
            en = beg + dim
            p.grad = grad[beg:en].data.clone()
            beg = en

    def safe_update(self, neg_safe_advantage):
        g1 = self.grads[:,0]
        g2 = -self.grads[:,1]
        phi = neg_safe_advantage.detach() - self.safety_threshold
        lmbd = F.relu((0.1 * phi - g1.dot(g2))/(g2.dot(g2)+1e-8))
        return g1 + lmbd * g2

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        action += np.random.normal(0, self.exploration_noise, size=action.shape)
        action *= self._action_scale.cpu().data.numpy()
        action += self._action_bias.cpu().data.numpy()
        return action

    def train(self, replay_buffer, batch_size=256):
        if self.safe_rl:
            return self.safe_train(replay_buffer, batch_size)

        self.total_it += 1

        # Sample replay buffer ("task" for multi-task learning)
        state, action, next_state, reward, not_done, task, ind = replay_buffer.sample(
            batch_size)

        next_state, reward, not_done, gammas = replay_buffer.n_step_return(self.n_step, ind, self.gamma)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * gammas * target_Q

        # Get current Q estimates
        action -= self._action_bias
        action /= self._action_scale  # to range of -1, 1
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
            F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = None
        # Delayed policy updates
        if self.total_it % self.update_actor_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

        actor_loss = actor_loss.item() if actor_loss is not None else None
        critic_loss = critic_loss.item()
        return self.grad_norm(self.actor), self.grad_norm(self.critic), actor_loss, critic_loss

    def safety_train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer ("task" for multi-task learning)
        state, action, next_state, reward, not_done, task, collision_reward, ind = replay_buffer.sample(
            batch_size)

        next_state, reward, not_done, gammas, collision_reward = replay_buffer.n_step_return(self.n_step, ind, self.gamma)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * gammas * target_Q

            # Compute the target safe Q value
            safe_target_Q1, safe_target_safe_Q2 = self.safe_critic_target(next_state, next_action)
            safe_target_Q = torch.min(safe_target_Q1, safe_target_Q2)
            safe_target_Q = collision_reward + not_done * gammas * safe_target_Q

        # Get current Q estimates
        action -= self._action_bias
        action /= self._action_scale  # to range of -1, 1
        current_Q1, current_Q2 = self.critic(state, action)
        safe_current_Q1, safe_current_Q2 = self.safe_critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
            F.mse_loss(current_Q2, target_Q)
        safe_critic_loss = F.mse_loss(safe_current_Q1, safe_target_Q) + \
            F.mse_loss(safe_current_Q2, safe_target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.safe_critic_optimizer.zero_grad()
        safe_critic_loss.backward()
        self.safe_critic_optimizer.step()

        actor_loss = None

        # Delayed policy updates
        if self.total_it % self.update_actor_freq == 0:

            # Compute actor losse
            action_now = self.actor(state)
            actor_loss = -self.critic.Q1(state, aciton_now).mean()
            safe_actor_loss = -self.safe_critic.Q1(state, action_now).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()

            grad_1 = torch.autograd.grad(actor_loss, self.actor.parameters())
            self.grad2vec(grad_1, 0)
            grad_2 = torch.autograd.grad(safe_actor_loss, self.actor.parameters())
            self.grad2vec(grad_1, 0)

            grad = self.safe_update(safe_actor_loss)
            self.vec2grad(grad)

            #actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

        actor_loss = actor_loss.item() if actor_loss is not None else None
        critic_loss = critic_loss.item()
        return self.grad_norm(self.actor), self.grad_norm(self.critic), actor_loss, critic_loss

    def grad_norm(self, model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2).item() if p.grad is not None else 0
            total_norm += param_norm ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def save(self, dir, filename):
        self.actor.to("cpu")
        with open(join(dir, filename + "_actor"), "wb") as f:
            pickle.dump(self.actor.state_dict(), f)
        with open(join(dir, filename + "_noise"), "wb") as f:
            pickle.dump(self.exploration_noise, f)
        self.actor.to(self.device)

    def load(self, dir, filename):
        with open(join(dir, filename + "_actor"), "rb") as f:
            self.actor.load_state_dict(pickle.load(f))
            self.actor_target = copy.deepcopy(self.actor)
        with open(join(dir, filename + "_noise"), "rb") as f:
            self.exploration_noise = pickle.load(f)


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device="cpu", safe_rl=False):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.mean, self.std = 0.0, 1.0

        self.safe_rl = safe_rl

        self.state = np.zeros((max_size, *state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, *state_dim))
        self.reward = np.zeros((max_size, 1))
        self.collision_reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.task = np.zeros((max_size, 1))

        self.device = device

    def add(self, state, action, next_state, reward, done, task, collision_reward=None):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward # (reward - 0.02478) / 6.499
        self.not_done[self.ptr] = 1. - done
        self.task[self.ptr] = task

        if self.safe_rl:
            assert collision_reward is not None, "collision reward should not be None"
            self.collision_reward[self.ptr] = collision_reward

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        if self.ptr == 1000:
            rew = self.reward[:1000]
            self.mean, self.std = rew.mean(), rew.std()
            if np.isclose(self.std, 0, 1e-2):
                self.mean, self.std = 0.0, 1.0

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        if self.safe_rl:
            return (
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device),
                torch.FloatTensor(self.task[ind]).to(self.device),
                torch.FloatTensor(self.collision_reward[ind]).to(self.device),
                ind)
        else:
            return (
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device),
                torch.FloatTensor(self.task[ind]).to(self.device),
                ind)

    def n_step_return(self, n_step, ind, gamma):
        reward = []
        not_done = []
        next_state = []
        gammas = []
        if self.safe_rl:
            collision_reward = []

        for i in ind:
            n = 0
            r = 0
            c = 0
            for _ in range(n_step):
                idx = (i + n) % self.size
                r += (self.reward[idx] - self.mean) / self.std * gamma ** n
                if self.safe_rl:
                    c += self.collision_reward[idx] * gamma ** n
                if not self.not_done[idx]:
                    break
                n = n + 1
            next_state.append(self.next_state[idx])
            not_done.append(self.not_done[idx])
            reward.append(r)
            gammas.append([gamma ** (n + 1)])
            if self.safe_rl:
                collision_reward.append(c)

        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        not_done = torch.FloatTensor(np.array(not_done)).to(self.device)
        reward = torch.FloatTensor(np.array(reward)).to(self.device)
        gammas = torch.FloatTensor(np.array(gammas)).to(self.device)
        if self.safe_rl:
            collision_reward = torch.FloatTensor(
                    np.array(collision_reward)).to(self.device)
        if self.safe_rl:
            return next_state, reward, not_done, gammas, collision_reward
        else:
            return next_state, reward, not_done, gammas
