import copy
import pickle
from os.path import join
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

from rl_algos.base_rl_algo import BaseRLAlgo


class EnsembleFC(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, ensemble_size: int, weight_decay: float = 0., bias: bool = True) -> None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


def init_weights(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t

    if type(m) == nn.Linear or isinstance(m, EnsembleFC):
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x


class EnsembleModel(nn.Module):
    def __init__(self, ensemble_size, hidden_size=200, learning_rate=1e-3, use_decay=False, device="cpu"):
        super(EnsembleModel, self).__init__()
        state_size, action_size = 724, 2
        self.hidden_size = hidden_size
        self.nn1 = EnsembleFC(state_size + action_size, hidden_size, ensemble_size, weight_decay=0.000025)
        self.nn2 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00005)
        self.nn3 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.nn4 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.use_decay = use_decay
        self.ensemble_size = ensemble_size
        self.device = device

        self.output_dim = state_size + 1  # predict one dense reward1
        # Add variance output
        self.nn5_sr = EnsembleFC(hidden_size, self.output_dim * 2, ensemble_size, weight_decay=0.0001)
        self.nn5_collision = EnsembleFC(hidden_size, 1, ensemble_size, weight_decay=0.0001)

        self.max_logvar = nn.Parameter((torch.ones((1, self.output_dim)).float() / 2).to(device), requires_grad=False)
        self.min_logvar = nn.Parameter((-torch.ones((1, self.output_dim)).float() * 10).to(device), requires_grad=False)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.apply(init_weights)
        self.swish = Swish()
        self.cel = torch.nn.CrossEntropyLoss()

    def forward(self, s, a, ret_log_var=False):
        s = s.view(s.shape[0], s.shape[1], -1)
        s = torch.cat([s, a], axis=-1)
        
        nn1_output = self.swish(self.nn1(s))
        nn2_output = self.swish(self.nn2(nn1_output))
        nn3_output = self.swish(self.nn3(nn2_output))
        nn4_output = self.swish(self.nn4(nn3_output))
        sr = self.nn5_sr(nn4_output)
        collision = torch.sigmoid(self.nn5_collision(nn4_output))

        mean = sr[:, :, :self.output_dim]

        logvar = self.max_logvar - F.softplus(self.max_logvar - sr[:, :, self.output_dim:])
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        if ret_log_var:
            return mean, logvar, collision
        else:
            return mean, torch.exp(logvar), collision

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
                # print(m.weight.shape)
                # print(m, decay_loss, m.weight_decay)
        return decay_loss

    def loss(self, mean, logvar, collision, labels, collision_labels, inc_var_loss=True):
        """
        mean, logvar: Ensemble_size x N x dim
        labels: N x dim
        """
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3
        inv_var = torch.exp(-logvar)
        if inc_var_loss:
            # Average over batch and dim, sum over ensembles.
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) * inv_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            c_loss = torch.mean(collision * (1 - collision_labels) + (1 - collision) * collision_labels, dim=1).view(-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss) + torch.sum(c_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            c_loss = torch.mean(collision * (1 - collision_labels) + (1 - collision) * collision_labels, dim=1).view(-1)
            total_loss = torch.sum(mse_loss) + torch.sum(c_loss)
        return total_loss, mse_loss + c_loss

    def train(self, loss):
        self.optimizer.zero_grad()
        loss += (0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar))
        # print('loss:', loss.item())
        if self.use_decay:
            loss += self.get_decay_loss()
        loss.backward()
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.grad.shape, torch.mean(param.grad), param.grad.flatten()[:5])
        self.optimizer.step()

    def sample(self, state, action, elite_model_idxes):
        state = torch.stack([state] * self.ensemble_size)
        action = torch.stack([action] * self.ensemble_size)
        means, vars, c = self(state, action)
        stds = torch.pow(vars, 0.5)
        samples = torch.normal(means, stds)

        num_models, batch_size, _ = means.shape
        model_idxes = torch.tensor(np.random.choice(elite_model_idxes, size=batch_size)).to(self.device)
        batch_idxes = torch.tensor(np.arange(0, batch_size)).to(self.device)

        sr = samples[model_idxes, batch_idxes]
        delta_state = sr[..., :-1]
        r = sr[..., -1]

        next_state = torch.cat([state[-1, :, :-1, :], (state[-1, :, -1, :] + delta_state)[:, None, :]], axis=1)
        reach_goal = ((torch.sum(next_state[:, -1, -4:-2].pow(2)) + 1) * 5) < 0.4
        collided = c[-1, :, 0] > 0.5

        r += reach_goal * 20
        r += (collided * -1.0)

        done = torch.logical_or(reach_goal, collided)
        return next_state, r.view(-1), done.view(-1)


class MBPO(BaseRLAlgo):
    def __init__(self, model, *args, model_update_epoch=5, n_simulated_update=5, n_real_update=5, ensemble_size=7, rollout_length=5, hold_ratio=0.1, elite_size=5, **kw_args):
        self.model = model
        self.model_update_epoch = model_update_epoch
        self.n_simulated_update = n_simulated_update
        self.rollout_length = rollout_length
        self.ensemble_size = ensemble_size
        self.n_real_update = n_real_update
        self.hold_ratio = hold_ratio
        self.elite_size = elite_size

        self.last_size = 0
        super().__init__(*args, **kw_args)

    def train_model(self, replay_buffer, batch_size=256):
        size = int(replay_buffer.size * (1 - self.hold_ratio))
        self._snapshots = {i: (None, 1e10) for i in range(self.ensemble_size)}
        for k in itertools.count():
            train_idx = np.vstack([np.random.permutation(size) for _ in range(self.ensemble_size)])
            for i in range(0, size, batch_size):
                idx = train_idx[:, i: i + batch_size]
                state, action, next_state, reward, not_done, gammas, collision_reward, index = replay_buffer.sample(batch_size, idx)
                action -= self._action_bias
                action /= self._action_scale
                done = 1 - not_done
                mean, logvar, c = self.model(state, action, ret_log_var=True)
                labels = torch.cat([next_state[:, :, -1, :] - state[:, :, -1, :], reward], axis=-1)
                loss, _ = self.model.loss(
                    mean, logvar, c, labels, collision_reward
                )
                self.model.train(loss)

            test_idx = np.vstack([np.array(range(size, replay_buffer.size)) for _ in range(self.ensemble_size)])
            test_loss = 0
            for i in range(0, replay_buffer.size - size, batch_size):
                idx = test_idx[:, i: i + batch_size]
                state, action, next_state, reward, not_done, gammas, collision_reward, index = replay_buffer.sample(batch_size, idx)
                action -= self._action_bias
                action /= self._action_scale
                done = 1 - not_done
                mean, logvar, c = self.model(state, action, ret_log_var=True)
                labels = torch.cat([next_state[:, :, -1, :] - state[:, :, -1, :], reward], axis=-1)
                _, loss = self.model.loss(
                    mean, logvar, c, labels, collision_reward, inc_var_loss=False
                )
                test_loss += loss
            test_loss = test_loss.detach().cpu().numpy()
            sorted_loss_idx = np.argsort(test_loss)
            self.elite_model_idxes = sorted_loss_idx[:self.elite_size].tolist()
            break_train = self._save_best(k, test_loss)
            if break_train:
                break
            print("Updating model %d, loss %.4f" %(k, np.mean(test_loss)), end="\r")
        return {
            "Model_loss": np.mean(test_loss)
        }

    def _save_best(self, epoch, holdout_losses):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                # self._save_state(i)
                updated = True
                # improvement = (best - current) / best

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self.model_update_epoch:
            return True
        else:
            return False

    def simulate_transition(self, replay_buffer, batch_size=256):
        idx = np.random.permutation(range(self.last_size, replay_buffer.size))
        state, action, next_state, reward, *_ = replay_buffer.sample(batch_size, idx)
        total_reward = torch.zeros(reward.shape).to(self.device)
        not_done = torch.ones(reward.shape).to(self.device)
        gammas = 1
        ss, aa, nss, rr, nds, gs = [], [], [], [], [], []
        with torch.no_grad():
            for i in range(self.rollout_length):
                action = self.actor_target(state)
                action += torch.randn_like(action, dtype=torch.float32) * self.exploration_noise  # exploration noise
                next_state, r, d = self.model.sample(state, action, self.elite_model_idxes)
                reward = r # self._get_reward(state, next_state)[:, None]
                not_done = torch.logical_not(d) # (1 - self._get_done(next_state)[:, None])
                gammas = (self.gamma * (not_done))

                ss.append(state)
                aa.append(action)
                nss.append(next_state)
                rr.append(reward)
                nds.append(not_done)
                gs.append(gammas)

                state = next_state

        return torch.cat(ss, dim=0), torch.cat(aa, dim=0), torch.cat(nss, dim=0), torch.cat(rr, dim=0), torch.cat(nds, dim=0), torch.cat(gs, dim=0)
    def train(self, replay_buffer, batch_size=256):
        # rl_loss_info = super().train(replay_buffer, batch_size)
        model_loss_info = self.train_model(replay_buffer, batch_size)

        simulated_rl_loss_infos = []
        rl_loss_infos = []
        for ii in range(self.n_simulated_update):
            print("Updating policy %d" %ii, end="\r")
            state, action, next_state, reward, not_done, gammas = self.simulate_transition(replay_buffer, batch_size)
            epoch_idx = torch.tensor(np.random.permutation(state.shape[0]))
            for i in range(0, state.shape[0], batch_size):
                idx = epoch_idx[i: i + batch_size]
                simulated_rl_loss_info = self.train_rl(state[idx], action[idx], next_state[idx], reward[idx], not_done[idx], gammas[idx], None)
                simulated_rl_loss_infos.append(simulated_rl_loss_info)

            if ii % (self.n_simulated_update // self.n_real_update + 1) == 0:
                rl_loss_info = super().train(replay_buffer, batch_size)
                rl_loss_infos.append(rl_loss_info)
        self.last_size = replay_buffer.size

        simulated_rl_loss_info = {}
        for k in simulated_rl_loss_infos[0].keys():
            simulated_rl_loss_info["simulated" + k] = np.mean([li[k] for li in simulated_rl_loss_infos if li[k] is not None])

        rl_loss_info = {}
        for k in rl_loss_infos[0].keys():
            rl_loss_info[k] = np.mean([li[k] for li in rl_loss_infos if li[k] is not None])

        loss_info = {}
        loss_info.update(rl_loss_info)
        loss_info.update(model_loss_info)
        loss_info.update(simulated_rl_loss_info)

        return loss_info
    
    def save(self, dir, filename):
        super().save(dir, filename)
        self.model.to("cpu")
        with open(join(dir, filename + "_model"), "wb") as f:
            pickle.dump(self.model.state_dict(), f)
        self.model.to(self.device)

    def load(self, dir, filename):
        super().load(dir, filename)
        with open(join(dir, filename + "_model"), "rb") as f:
            self.model.load_state_dict(pickle.load(f))


class MBPORLAlgo(BaseRLAlgo):
    def __init__(self, model, model_optm, *args, model_update_per_step=5, n_simulated_update=5, **kw_args):
        self.model = model
        self.model_optimizer = model_optm
        self.model_update_per_step = model_update_per_step
        self.n_simulated_update = n_simulated_update
        self.loss_function = nn.MSELoss()
        self.start_idx = None
        super().__init__(*args, **kw_args)

    def train_model(self, replay_buffer, batch_size=256):
        state, action, next_state, reward, not_done, *_ = replay_buffer.sample(batch_size)
        action -= self._action_bias
        action /= self._action_scale
        done = 1 - not_done
        if self.model.deterministic:
            pred_next_state, r, d = self.model(state, action)
            state_loss = self.loss_function(pred_next_state, next_state[:, -1, :])
        else:
            pred_next_state_mean_var, r, d = self.model(state, action)
            mean = pred_next_state_mean_var[..., self.model.state_dim // 2:]
            logvar = pred_next_state_mean_var[..., :self.model.state_dim // 2]
            recon_dist = Normal(mean, torch.exp(logvar))
            state_loss = -recon_dist.log_prob(next_state[:, -1, :]).sum(dim=-1).mean()  # nll loss
        
        # pred_reward = self._get_reward(state, pred_next_state[:, None, :])
        # pred_done = self._get_done(pred_next_state[:, None, :])
        reward_loss = self.loss_function(r, reward)
        # done_loss = self.loss_function(d, done.view(-1))
        done_loss = F.binary_cross_entropy(d, done)

        loss = state_loss + reward_loss + done_loss

        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()
        return {
            "Model_loss": loss.item(),
            "Model_grad_norm": self.grad_norm(self.model)
        }

    def simulate_transition(self, replay_buffer, batch_size=256):
        # MBPO only branch from the states visited by the current policy, or newly collected states
        state, _, next_state, reward, *_ = replay_buffer.sample(batch_size, start_idx = self.start_idx)
        total_reward = torch.zeros(reward.shape).to(self.device)
        not_done = torch.ones(reward.shape).to(self.device)
        gammas = 1
        with torch.no_grad():
            for i in range(self.n_step):
                next_action = self.select_action(state, to_cpu=False)
                if i == 0:
                    action = next_action
                next_state, r, d = self.model.sample(state, next_action)
                reward = r # self._get_reward(state, next_state)[:, None]
                not_done *= (1 - d) # (1 - self._get_done(next_state)[:, None])
                gammas *= self.gamma ** (not_done)
                reward = (reward - replay_buffer.mean) / replay_buffer.std  # reward normalization 
                total_reward = reward + total_reward * gammas

        return state, action, next_state, reward, not_done, gammas

    def train(self, replay_buffer, batch_size=256):
        # rl_loss_info = super().train(replay_buffer, batch_size)  # train MBPO purely on model-generated data
        for _ in range(self.model_update_per_step):
            model_loss_info = self.train_model(replay_buffer, batch_size)

        simulated_rl_loss_infos = []
        transitions = []
        # sample all the transition samples before update the policy
        for _ in range(self.n_simulated_update):
            transitions.append(self.simulate_transition(replay_buffer, batch_size))
        for state, action, next_state, reward, not_done, gammas in transitions:
            simulated_rl_loss_info = self.train_rl(state, action, next_state, reward, not_done, gammas)
            simulated_rl_loss_infos.append(simulated_rl_loss_info)
        self.start_idx = replay_buffer.ptr

        simulated_rl_loss_info = {}
        for k in simulated_rl_loss_infos[0].keys():
            simulated_rl_loss_info["simulated" + k] = np.mean([li[k] for li in simulated_rl_loss_infos if li[k] is not None])

        loss_info = {}
        # loss_info.update(rl_loss_info)
        loss_info.update(model_loss_info)
        loss_info.update(simulated_rl_loss_info)

        return loss_info
    
    def save(self, dir, filename):
        super().save(dir, filename)
        self.model.to("cpu")
        with open(join(dir, filename + "_model"), "wb") as f:
            pickle.dump(self.model.state_dict(), f)
        self.model.to(self.device)

    def load(self, dir, filename):
        super().load(dir, filename)
        with open(join(dir, filename + "_model"), "rb") as f:
            self.model.load_state_dict(pickle.load(f))


class SMCPRLAlgo(BaseRLAlgo):
    def __init__(self, model, model_optm, *args, horizon=5, num_particle=1024, model_update_per_step=5, **kw_args):
        self.model = model
        self.model_optimizer = model_optm
        self.horizon = horizon
        self.num_particle = num_particle
        self.model_update_per_step = model_update_per_step
        self.loss_function = nn.MSELoss()
        super().__init__(*args, **kw_args)

    def train_model(self, replay_buffer, batch_size=256):
        state, action, next_state, reward, not_done, _, _ = replay_buffer.sample(batch_size)
        action -= self._action_bias
        action /= self._action_scale
        done = 1 - not_done
        if self.model.deterministic:
            pred_next_state, r, d = self.model(state, action)
            state_loss = self.loss_function(pred_next_state, next_state[:, -1, :])
        else:
            pred_next_state_mean_var, r, d = self.model(state, action)
            mean = pred_next_state_mean_var[..., self.model.state_dim // 2:]
            logvar = pred_next_state_mean_var[..., :self.model.state_dim // 2]
            recon_dist = Normal(mean, torch.exp(logvar))
            state_loss = -recon_dist.log_prob(next_state[:, -1, :]).sum(dim=-1).mean()  # nll loss
        
        # pred_reward = self._get_reward(state, pred_next_state[:, None, :])
        # pred_done = self._get_done(pred_next_state[:, None, :])
        reward_loss = self.loss_function(r, reward)
        # done_loss = self.loss_function(d, done.view(-1))
        done_loss = F.binary_cross_entropy(d, done)

        loss = state_loss + reward_loss + done_loss

        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()
        return {
            "Model_loss": loss.item(),
            "Model_grad_norm": self.grad_norm(self.model)
        }
        
    def train(self, replay_buffer, batch_size=256):
        rl_loss_info = super().train(replay_buffer, batch_size)
        for _ in range(self.model_update_per_step):
            model_loss_info = self.train_model(replay_buffer, batch_size)
        
        loss_info = {}
        loss_info.update(rl_loss_info)
        loss_info.update(model_loss_info)

        return loss_info

    def select_action(self, state):
        if self.exploration_noise >= 0:
            assert len(state.shape) == 2, "does not support batched action selection!"
            state = torch.FloatTensor(state).to(self.device)[None, ...]  # (batch_size=1, history_length, 723)
            s = state.repeat(self.num_particle, 1, 1).clone()
            r = 0
            gamma = torch.zeros((self.num_particle, 1)).to(self.device)
            with torch.no_grad():
                for i in range(self.horizon):
                    # Sample action with policy
                    a = self.actor(s)
                    a += torch.randn_like(a, dtype=torch.float32) * self.exploration_noise
                    if i == 0:
                        a0 = a
                        
                    # simulate trajectories
                    ns, r, d = self.model.sample(s, a)
                    r += r * gamma # self._get_reward(s, ns)[:, None] * gamma
                    gamma *= (1 - d) # (1 - self._get_done(ns)[:, None])
                    s = ns
                q = self.critic.Q1(ns, a)
                r += q * gamma
                
                logit_r = F.softmax(r, -1).view(-1)
                n = Categorical(logit_r).sample()
                a = a0[n]
            return a.cpu().data.numpy()
        else: # deploy the policy only when self.exploration_noise = 0 
            return super().select_action(state)
    
    def save(self, dir, filename):
        super().save(dir, filename)
        self.model.to("cpu")
        with open(join(dir, filename + "_model"), "wb") as f:
            pickle.dump(self.model.state_dict(), f)
        self.model.to(self.device)

    def load(self, dir, filename):
        super().load(dir, filename)
        with open(join(dir, filename + "_model"), "rb") as f:
            self.model.load_state_dict(pickle.load(f))