import os
import sys

import copy
from datetime import datetime
from multiprocessing import set_start_method
import torch.multiprocessing as torch_mp
import multiprocessing as mp
import queue
from time import sleep

sys.path.append('../')
from insomnia.agents.d4pg_learner import LearnerD4PG
from insomnia.agents.d4pg_agent import Agent
from insomnia.numeric_models import d4pg
from insomnia.replay_buffers.prioritized_buffer import create_replay_buffer
from insomnia.utils import empty_torch_queue

try:
    set_start_method('spawn')
except RuntimeError:
    pass

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def sampler_worker(replay_queue, batch_queue, replay_priorities_queue, training_on,
                   global_episode, update_step, log_dir, batch_size, mem_size):
    """
    Function that transfers replay to the buffer and batches from buffer to the queue.
    Args:
        config:
        replay_queue:
        batch_queue:
        training_on:
        global_episode:
        log_dir:
    """

    # Create replay buffer
    replay_buffer = create_replay_buffer(mem_size)

    while training_on.value:
        # (1) Transfer replays to global buffer
        n = replay_queue.qsize()
        for _ in range(n):
            replay = replay_queue.get()
            replay_buffer.add(*replay)

        # (2) Transfer batch of replay from buffer to the batch_queue
        if len(replay_buffer) < batch_size:
            continue

        try:
            inds, weights = replay_priorities_queue.get_nowait()
            replay_buffer.update_priorities(inds, weights)
        except queue.Empty:
            pass

        try:
            batch = replay_buffer.sample(batch_size)
            batch_queue.put_nowait(batch)
        except:
            sleep(0.1)
            continue

    empty_torch_queue(batch_queue)
    print("Stop sampler worker.")


def learner_worker(training_on, policy, target_policy_net, learner_w_queue, replay_priority_queue,
                   batch_queue, update_step, alpha, beta, input_dims, n_actions, fc1_dims, fc2_dims, name, v_min, v_max):
    learner = LearnerD4PG(policy, target_policy_net, learner_w_queue,
                          alpha, beta, input_dims, n_actions, fc1_dims, fc2_dims, name, v_min, v_max)
    learner.run(training_on, batch_queue, replay_priority_queue, update_step)


def agent_worker(policy, learner_w_queue, global_episode, i, agent_type,
                 experiment_dir, training_on, replay_queue, update_step, n_actions):
    agent = Agent(policy=policy,
                  global_episode=global_episode,
                  n_actions=n_actions,
                  n_agent=i,
                  agent_type=agent_type,
                  log_dir=experiment_dir)
    agent.run(training_on, replay_queue, learner_w_queue, update_step)

def train(self):
    alpha = 0.0005
    beta = 0.0005
    input_dims = [8]


    batch_queue_size = 64
    n_agents = 4

    # Create directory for experiment
    experiment_dir = ''
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # Data structures
    processes = []
    replay_queue = mp.Queue(maxsize=64)
    training_on = mp.Value('i', 1)
    update_step = mp.Value('i', 0)
    global_episode = mp.Value('i', 0)
    learner_w_queue = torch_mp.Queue(maxsize=n_agents)
    replay_priorities_queue = mp.Queue(maxsize=64)

    # Data sampler
    batch_queue = mp.Queue(maxsize=batch_queue_size)
    p = torch_mp.Process(target=sampler_worker,
                         args=(replay_queue, batch_queue, replay_priorities_queue, training_on,
                               global_episode, update_step, experiment_dir))
    processes.append(p)

    # Learner (neural net training process)
    target_policy_net = d4pg.ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims, n_actions, name)
    policy_net = copy.deepcopy(target_policy_net)
    policy_net_cpu = d4pg.ActorNetwork(input_dims, n_actions,
                                   config['dense_size'])
    target_policy_net.share_memory()

    p = torch_mp.Process(target=learner_worker,
                         args=(config, training_on, policy_net, target_policy_net, learner_w_queue,
                               replay_priorities_queue, batch_queue, update_step, experiment_dir))
    processes.append(p)

    # Single agent for exploitation
    p = torch_mp.Process(target=agent_worker,
                         args=(config, target_policy_net, None, global_episode, 0, "exploitation", experiment_dir,
                               training_on, replay_queue, update_step))
    processes.append(p)

    # Agents (exploration processes)
    for i in range(1, n_agents):
        p = torch_mp.Process(target=agent_worker,
                             args=(config, policy_net_cpu, learner_w_queue, global_episode, i, "exploration",
                                   experiment_dir,
                                   training_on, replay_queue, update_step))
        processes.append(p)

    for p in processes:
        p.start()
    for p in processes:
        p.join()



'''

agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env,
              batch_size=64, layer1_size=400, layer2_size=300, n_actions=2)

agent = Agent

# agent.load_models()
np.random.seed(0)

score_history = []
for i in range(2000):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        act = agent.choose_action(obs)
        new_state, reward, done, info = env.step(act)

        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state
        env.render()
    score_history.append(score)

    if i % 25 == 0:
        agent.save_models()

    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))

filename = 'LunarLander-alpha000025-beta00025-400-300.png'
plt.plot(score_history, label='learning-curve')
plt.savefig(filename)
plt.show()
'''


