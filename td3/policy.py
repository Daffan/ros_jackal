import torch
import numpy as np
from copy import deepcopy
from typing import Any, Dict, Tuple, Union, Optional

from torch.distributions import Independent, Normal
from tianshou.policy import BasePolicy
from tianshou.exploration import BaseNoise, GaussianNoise
from tianshou.data import Batch, ReplayBuffer, to_torch_as


class DDPGPolicy(BasePolicy):
    """Implementation of Deep Deterministic Policy Gradient. arXiv:1509.02971.
    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic: the critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic_optim: the optimizer for critic
        network.
    :param action_range: the action range (minimum, maximum).
    :type action_range: Tuple[float, float]
    :param float tau: param for soft update of the target network, defaults to
        0.005.
    :param float gamma: discount factor, in [0, 1], defaults to 0.99.
    :param BaseNoise exploration_noise: the exploration noise,
        add to the action, defaults to ``GaussianNoise(sigma=0.1)``.
    :param bool reward_normalization: normalize the reward to Normal(0, 1),
        defaults to False.
    :param bool ignore_done: ignore the done flag while training the policy,
        defaults to False.
    :param int estimation_step: greater than 1, the number of steps to look
        ahead.
    .. seealso::
        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        actor: Optional[torch.nn.Module],
        actor_optim: Optional[torch.optim.Optimizer],
        critic: Optional[torch.nn.Module],
        critic_optim: Optional[torch.optim.Optimizer],
        action_range: Tuple[float, float],
        tau: float = 0.005,
        gamma: float = 0.99,
        exploration_noise: Optional[BaseNoise] = GaussianNoise(sigma=0.1),
        reward_normalization: bool = False,
        ignore_done: bool = False,
        estimation_step: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if actor is not None and actor_optim is not None:
            self.actor: torch.nn.Module = actor
            self.actor_old = deepcopy(actor)
            self.actor_old.eval()
            self.actor_optim: torch.optim.Optimizer = actor_optim
        if critic is not None and critic_optim is not None:
            self.critic: torch.nn.Module = critic
            self.critic_old = deepcopy(critic)
            self.critic_old.eval()
            self.critic_optim: torch.optim.Optimizer = critic_optim
        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        self._tau = tau
        assert 0.0 <= gamma <= 1.0, "gamma should be in [0, 1]"
        self._gamma = gamma
        self._noise = exploration_noise
        self._range = action_range
        # self._action_bias = (action_range[0] + action_range[1]) / 2.0
        # force policy to center at init parameters
        # self._action_bias = torch.tensor(np.array([0.5, 1.57, 6, 20, 0.75, 1, 0.3]))
        # it is only a little difference to use GaussianNoise
        # self.noise = OUNoise()
        self._rm_done = ignore_done
        self._rew_norm = reward_normalization
        assert estimation_step > 0, "estimation_step should be greater than 0"
        self._n_step = estimation_step
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._action_scale = torch.tensor((action_range[1] - action_range[0]) / 2.0, device=self.device)
        self._action_bias = torch.tensor((action_range[0] + action_range[1]) / 2.0, device=self.device)

    def set_exp_noise(self, noise: Optional[BaseNoise]) -> None:
        """Set the exploration noise."""
        self._noise = noise

    def train(self, mode: bool = True) -> "DDPGPolicy":
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.actor.train(mode)
        self.critic.train(mode)
        return self

    def sync_weight(self) -> None:
        """Soft-update the weight for the target network."""
        for o, n in zip(self.actor_old.parameters(), self.actor.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(
            self.critic_old.parameters(), self.critic.parameters()
        ):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    def _target_q(
        self, buffer: ReplayBuffer, indice: np.ndarray
    ) -> torch.Tensor:
        batch = buffer[indice]
        batch.to_torch(dtype=torch.float32, device=self.device)  # batch.obs_next: s_{t+n}
        with torch.no_grad():
            target_q = self.critic_old(
                batch.obs_next,
                self(batch, model='actor_old', input='obs_next').act)
        return target_q

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        if self._rm_done:
            batch.done = batch.done * 0.0
        batch = self.compute_nstep_return(
            batch, buffer, indice, self._target_q,
            self._gamma, self._n_step, self._rew_norm)
        return batch

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "actor",
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.
        :return: A :class:`~tianshou.data.Batch` which has 2 keys:
            * ``act`` the action.
            * ``state`` the hidden state.
        .. seealso::
            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        model = getattr(self, model)
        obs = batch[input]
        actions, h = model(obs, state=state, info=batch.info)
        if self._noise and not self.updating:
            actions += to_torch_as(self._noise(actions.shape), actions)
        actions *= self._action_scale
        actions += self._action_bias
        for i, (low, high) in enumerate(zip(self._range[0], self._range[1])):
            actions[:,i] = actions.clone()[:,i].clamp(low, high)
        # actions = actions.clamp(self._range[0], self._range[1])
        return Batch(act=actions, state=h)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        weight = batch.pop("weight", 1.0)
        current_q = self.critic(batch.obs, batch.act).flatten()
        target_q = batch.returns.flatten()
        td = current_q - target_q
        critic_loss = (td.pow(2) * weight).mean()
        batch.weight = td  # prio-buffer
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        action = self(batch).act
        actor_loss = -self.critic(batch.obs, action).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.sync_weight()
        return {
            "loss/actor": actor_loss.item(),
            "loss/critic": critic_loss.item(),
        }

class TD3Policy(DDPGPolicy):
    """Implementation of TD3, arXiv:1802.09477.
    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic1: the first critic network. (s, a -> Q(s,
        a))
    :param torch.optim.Optimizer critic1_optim: the optimizer for the first
        critic network.
    :param torch.nn.Module critic2: the second critic network. (s, a -> Q(s,
        a))
    :param torch.optim.Optimizer critic2_optim: the optimizer for the second
        critic network.
    :param action_range: the action range (minimum, maximum).
    :type action_range: Tuple[float, float]
    :param float tau: param for soft update of the target network, defaults to
        0.005.
    :param float gamma: discount factor, in [0, 1], defaults to 0.99.
    :param float exploration_noise: the exploration noise, add to the action,
        defaults to ``GaussianNoise(sigma=0.1)``
    :param float policy_noise: the noise used in updating policy network,
        default to 0.2.
    :param int update_actor_freq: the update frequency of actor network,
        default to 2.
    :param float noise_clip: the clipping range used in updating policy
        network, default to 0.5.
    :param bool reward_normalization: normalize the reward to Normal(0, 1),
        defaults to False.
    :param bool ignore_done: ignore the done flag while training the policy,
        defaults to False.
    .. seealso::
        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1: torch.nn.Module,
        critic1_optim: torch.optim.Optimizer,
        critic2: torch.nn.Module,
        critic2_optim: torch.optim.Optimizer,
        action_range: Tuple[float, float],
        tau: float = 0.005,
        gamma: float = 0.99,
        exploration_noise: Optional[BaseNoise] = GaussianNoise(sigma=0.1),
        policy_noise: float = 0.2,
        update_actor_freq: int = 2,
        noise_clip: float = 0.5,
        reward_normalization: bool = False,
        ignore_done: bool = False,
        estimation_step: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(actor, actor_optim, None, None, action_range,
                         tau, gamma, exploration_noise, reward_normalization,
                         ignore_done, estimation_step, **kwargs)
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic1_optim = critic1_optim
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()
        self.critic2_optim = critic2_optim
        self._policy_noise = policy_noise
        self._freq = update_actor_freq
        self._noise_clip = noise_clip
        self._cnt = 0
        self._last = 0

    def train(self, mode: bool = True) -> "TD3Policy":
        self.training = mode
        self.actor.train(mode)
        self.critic1.train(mode)
        self.critic2.train(mode)
        return self

    def sync_weight(self) -> None:
        for o, n in zip(self.actor_old.parameters(), self.actor.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(
            self.critic1_old.parameters(), self.critic1.parameters()
        ):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(
            self.critic2_old.parameters(), self.critic2.parameters()
        ):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    def _target_q(
        self, buffer: ReplayBuffer, indice: np.ndarray
    ) -> torch.Tensor:
        batch = buffer[indice]
        batch.to_torch(dtype=torch.float32, device=self.device)  # batch.obs: s_{t+n}
        with torch.no_grad():
            a_ = self(batch, model="actor_old", input="obs_next").act
            dev = a_.device
            noise = torch.randn(size=a_.shape, device=dev) * self._policy_noise
            if self._noise_clip > 0.0:
                noise = noise.clamp(-self._noise_clip, self._noise_clip)
            a_ += noise
            a_ *= self._action_scale
            a_ += self._action_bias
            for i, (low, high) in enumerate(zip(self._range[0], self._range[1])):
                a_[:,i] = a_[:,i].clamp(low, high)
            #a_ = a_.clamp(self._range[0], self._range[1])
            target_q = torch.min(
                self.critic1_old(batch.obs_next, a_),
                self.critic2_old(batch.obs_next, a_))
        return target_q

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        weight = batch.pop("weight", 1.0)
        # critic 1
        current_q1 = self.critic1(batch.obs, batch.act).flatten()
        target_q = batch.returns.flatten()
        td1 = current_q1 - target_q
        critic1_loss = (td1.pow(2) * weight).mean()
        # critic1_loss = F.mse_loss(current_q1, target_q)
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()
        # critic 2
        current_q2 = self.critic2(batch.obs, batch.act).flatten()
        td2 = current_q2 - target_q
        critic2_loss = (td2.pow(2) * weight).mean()
        # critic2_loss = F.mse_loss(current_q2, target_q)
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()
        batch.weight = (td1 + td2) / 2.0  # prio-buffer
        if self._cnt % self._freq == 0:
            actor_loss = -self.critic1(
                batch.obs, self(batch, eps=0.0).act).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self._last = actor_loss.item()
            self.actor_optim.step()
            self.sync_weight()
        self._cnt += 1
        return {
            "loss/actor": self._last,
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }

class SACPolicy(DDPGPolicy):
    """Implementation of Soft Actor-Critic. arXiv:1812.05905.
    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic1: the first critic network. (s, a -> Q(s,
        a))
    :param torch.optim.Optimizer critic1_optim: the optimizer for the first
        critic network.
    :param torch.nn.Module critic2: the second critic network. (s, a -> Q(s,
        a))
    :param torch.optim.Optimizer critic2_optim: the optimizer for the second
        critic network.
    :param action_range: the action range (minimum, maximum).
    :type action_range: Tuple[float, float]
    :param float tau: param for soft update of the target network, defaults to
        0.005.
    :param float gamma: discount factor, in [0, 1], defaults to 0.99.
    :param (float, torch.Tensor, torch.optim.Optimizer) or float alpha: entropy
        regularization coefficient, default to 0.2.
        If a tuple (target_entropy, log_alpha, alpha_optim) is provided, then
        alpha is automatatically tuned.
    :param bool reward_normalization: normalize the reward to Normal(0, 1),
        defaults to False.
    :param bool ignore_done: ignore the done flag while training the policy,
        defaults to False.
    :param BaseNoise exploration_noise: add a noise to action for exploration,
        defaults to None. This is useful when solving hard-exploration problem.
    .. seealso::
        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1: torch.nn.Module,
        critic1_optim: torch.optim.Optimizer,
        critic2: torch.nn.Module,
        critic2_optim: torch.optim.Optimizer,
        action_range: Tuple[float, float],
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: Union[
            float, Tuple[float, torch.Tensor, torch.optim.Optimizer]
        ] = 0.2,
        reward_normalization: bool = False,
        ignore_done: bool = False,
        estimation_step: int = 1,
        exploration_noise: Optional[BaseNoise] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(None, None, None, None, action_range, tau, gamma,
                         exploration_noise, reward_normalization, ignore_done,
                         estimation_step, **kwargs)
        self.actor, self.actor_optim = actor, actor_optim
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic1_optim = critic1_optim
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()
        self.critic2_optim = critic2_optim

        self._is_auto_alpha = False
        self._alpha: Union[float, torch.Tensor]
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self._alpha_optim = alpha
            assert alpha[1].shape == torch.Size([1]) and alpha[1].requires_grad
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha

        self.__eps = np.finfo(np.float32).eps.item()

    def train(self, mode: bool = True) -> "SACPolicy":
        self.training = mode
        self.actor.train(mode)
        self.critic1.train(mode)
        self.critic2.train(mode)
        return self

    def sync_weight(self) -> None:
        for o, n in zip(
            self.critic1_old.parameters(), self.critic1.parameters()
        ):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(
            self.critic2_old.parameters(), self.critic2.parameters()
        ):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    def forward(  # type: ignore
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        obs = batch[input]
        logits, h = self.actor(obs, state=state, info=batch.info)
        assert isinstance(logits, tuple)
        dist = Independent(Normal(*logits), 1)
        x = dist.rsample()
        y = torch.tanh(x)
        act = y * self._action_scale + self._action_bias
        y = self._action_scale * (1 - y.pow(2)) + self.__eps
        log_prob = dist.log_prob(x).unsqueeze(-1)
        log_prob = log_prob - torch.log(y).sum(-1, keepdim=True)
        if self._noise is not None and not self.updating:
            act += to_torch_as(self._noise(act.shape), act)
        #act = act.clamp(self._range[0], self._range[1])
        for i, (low, high) in enumerate(zip(self._range[0], self._range[1])):
            act[:,i] = act.clone()[:,i].clamp(low, high)
        return Batch(
            logits=logits, act=act, state=h, dist=dist, log_prob=log_prob)

    def _target_q(
        self, buffer: ReplayBuffer, indice: np.ndarray
    ) -> torch.Tensor:
        batch = buffer[indice]
        batch.to_torch(dtype=torch.float32, device=self.device)  # batch.obs: s_{t+n}
        with torch.no_grad():
            obs_next_result = self(batch, input='obs_next')
            a_ = obs_next_result.act
            batch.act = to_torch_as(batch.act, a_)
            target_q = torch.min(
                self.critic1_old(batch.obs_next, a_),
                self.critic2_old(batch.obs_next, a_),
            ) - self._alpha * obs_next_result.log_prob
        return target_q

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        weight = batch.pop("weight", 1.0)

        # critic 1
        current_q1 = self.critic1(batch.obs, batch.act).flatten()
        target_q = batch.returns.flatten()
        td1 = current_q1 - target_q
        critic1_loss = (td1.pow(2) * weight).mean()
        # critic1_loss = F.mse_loss(current_q1, target_q)
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        # critic 2
        current_q2 = self.critic2(batch.obs, batch.act).flatten()
        td2 = current_q2 - target_q
        critic2_loss = (td2.pow(2) * weight).mean()
        # critic2_loss = F.mse_loss(current_q2, target_q)
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()
        batch.weight = (td1 + td2) / 2.0  # prio-buffer

        # actor
        obs_result = self(batch)
        a = obs_result.act
        current_q1a = self.critic1(batch.obs, a).flatten()
        current_q2a = self.critic2(batch.obs, a).flatten()
        actor_loss = (self._alpha * obs_result.log_prob.flatten()
                      - torch.min(current_q1a, current_q2a)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_prob = obs_result.log_prob.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_prob).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()

        self.sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }
        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()  # type: ignore

        return result