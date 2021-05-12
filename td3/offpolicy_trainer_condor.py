import time
import tqdm
import yaml
import numpy as np
import collections
from tensorboardX import SummaryWriter
from typing import Dict, List, Union, Callable, Optional

from tianshou.data import Collector
from tianshou.policy import BasePolicy
from tianshou.utils import tqdm_config, MovAvg
from tianshou.trainer import test_episode, gather_info

def offpolicy_trainer_condor(
        policy: BasePolicy,
        train_collector,
        max_epoch: int,
        step_per_epoch: int,
        collect_per_step: int,
        batch_size: int,
        update_per_step: int = 1,
        train_fn: Optional[Callable[[int], None]] = None,
        writer: Optional[SummaryWriter] = None,
        log_interval: int = 100,
) -> int:
    """A wrapper for off-policy trainer procedure. The ``step`` in trainer
    means a policy network update.
    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy`
        class.
    :param train_collector: the collector used for training.
    :type train_collector: :class:`~tianshou.data.Collector`
    :param test_collector: the collector used for testing.
    :type test_collector: :class:`~tianshou.data.Collector`
    :param int max_epoch: the maximum of epochs for training. The training
        process might be finished before reaching the ``max_epoch``.
    :param int step_per_epoch: the number of step for updating policy network
        in one epoch.
    :param int collect_per_step: the number of frames the collector would
        collect before the network update. In other words, collect some frames
        and do some policy network update.
    :param episode_per_test: the number of episodes for one policy evaluation.
    :param int batch_size: the batch size of sample data, which is going to
        feed in the policy network.
    :param int update_per_step: the number of times the policy network would
        be updated after frames are collected, for example, set it to 256 means
        it updates policy 256 times once after ``collect_per_step`` frames are
        collected.
    :param function train_fn: a function receives the current number of epoch
        index and performs some operations at the beginning of training in this
        epoch.
    :param torch.utils.tensorboard.SummaryWriter writer: a TensorBoard
        SummaryWriter.
    :param int log_interval: the log interval of the writer.
    :return: See :func:`~tianshou.trainer.gather_info`.
    """
    def int_to_BARN_world(w):
        return "BARN/world_%d.world" %(w)

    with open("configs/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    worlds = [int_to_BARN_world(w) if isinstance(w, int) else w for w in config["condor_config"]["worlds"]]

    global_step = 0
    update_step = 0
    best_epoch, best_reward = -1, -1.
    stat = {}
    start_time = time.time()
    results = collections.deque(maxlen=100)
    world_results = [collections.deque(maxlen=10) for _ in range(len(worlds))]
    world_count = [1] * len(worlds)
    world_pcount = [1] * len(worlds)
    for epoch in range(1, 1 + max_epoch):
        # train
        policy.train()
        if train_fn:
            train_fn(epoch)
        with tqdm.tqdm(total=step_per_epoch, desc=f'Epoch #{epoch}',
                       **tqdm_config) as t:
            while t.n < t.total:
                assert train_collector.policy == policy
                result = train_collector.collect(n_step=collect_per_step)
                world_pcount = world_count.copy()
                for i, world in enumerate(result["world"]):
                    w = worlds.index(world)
                    world_results[w].append({"ep_rew": result["ep_rew"][i],\
                                          "ep_len": result["ep_len"][i],\
                                          "success": result["success"][i],\
                                          "global_step": global_step})
                    world_count[w] += 1
                for world in worlds:
                    w = worlds.index(world) 
                    if world_count[w] // 10 > world_pcount[w] // 10:  # log per 100 episode 
                        for k in world_results[w][0].keys():
                            writer.add_scalar('%s' %(world) + "/" + k,
                                              np.mean([r[k] for r in world_results[w]]),
                                              global_step=world_count[w])
                n_ep = len(result["success"])
                result = [{"ep_rew":result["ep_rew"][i],\
                           "ep_len":result["ep_len"][i],\
                           "success":result["success"][i]}\
                           for i in range(n_ep)]
                results.extend(result)
                data = {"n_ep": n_ep}
                n_step = sum([r["ep_len"] for r in result])
                global_step += n_step 
                n_step = np.clip(n_step, 10, 5000)
                for i in range(update_per_step * min(n_step // collect_per_step, t.total - t.n)):
                # for i in range(update_per_step):# * min(n_step // collect_per_step, t.total - t.n)):
                    losses = policy.update(batch_size, train_collector.buffer)
                    update_step += 1
                    if len(result) > 0:
                        for k in result[0].keys():
                            data[k] = f"{np.mean([r[k] for r in result]):.2f}"
                            if writer and update_step % log_interval == 0:
                                writer.add_scalar('train/' + k, np.mean([r[k] for r in results]),
                                                  global_step=global_step)
                    for k in losses.keys():
                        if stat.get(k) is None:
                            stat[k] = MovAvg()
                        stat[k].add(losses[k])
                        data[k] = f'{stat[k].get():.6f}'
                        if writer and update_step % log_interval == 0:
                            writer.add_scalar(
                                k, stat[k].get(), global_step=update_step)
                    try:
                        data['exp_noise'] = policy._noise._sigma
                    except:
                        data['exp_noise'] = policy._noise
                    t.update(1)
                    t.set_postfix(**data)
            if t.n <= t.total:
                t.update()
    return global_step
