from collections import deque

import numpy as np
import torch
import gym

from insomnia.action_noise import GaussianActionNoise
from insomnia.utils import empty_torch_queue


class Agent(object):

    def __init__(self, policy, global_episode, n_actions,  n_agent=0, agent_type='exploration', log_dir=''):
        print(f"Initializing agent {n_agent}...")
        self.n_agent = n_agent
        self.agent_type = agent_type
        self.max_steps = 1000
        self.num_episode_save = 100
        self.global_episode = global_episode
        self.local_episode = 0
        self.log_dir = log_dir
        self.n_steps_return = 5
        self.gamma = 0.99

        # Create environment
        self.env = gym.make('Pendulum-v0')
        self.noise = GaussianActionNoise(mu=np.zeros(n_actions))

        self.actor = policy
        print("Agent ", n_agent, self.actor.device)

    def update_actor_learner(self, learner_w_queue, training_on):
        """Update local actor to the actor from learner. """
        if not training_on.value:
            return
        try:
            source = learner_w_queue.get_nowait()
        except:
            return
        target = self.actor
        for target_param, source_param in zip(target.parameters(), source):
            w = torch.tensor(source_param).float()
            target_param.data.copy_(w)
        del source

    def run(self, training_on, replay_queue, learner_w_queue, update_step):
        # Initialise deque buffer to store experiences for N-step returns
        self.exp_buffer = deque()

        best_reward = -float("inf")
        rewards = []
        while training_on.value:
            episode_reward = 0
            num_steps = 0
            self.local_episode += 1
            self.global_episode.value += 1
            self.exp_buffer.clear()

            if self.local_episode % 100 == 0:
                print(f"Agent: {self.n_agent}  episode {self.local_episode}")

            state = self.env.reset()

            done = False
            while not done:
                mu = self.actor.forward(state).to(self.actor.device)
                mu_prime = mu + torch.tensor(self.noise(), dtype=torch.float).to(self.actor.device)
                action = mu_prime.squeeze(0)
                next_state, reward, done, _ = self.env.step(action)

                episode_reward += reward

                self.exp_buffer.append((state, action, reward))

                # We need at least N steps in the experience buffer before we can compute Bellman
                # rewards and add an N-step experience to replay memory
                if len(self.exp_buffer) >= self.n_steps_return:
                    state_0, action_0, reward_0 = self.exp_buffer.popleft()
                    discounted_reward = reward_0
                    _gamma = self.gamma
                    for (_, _, r_i) in self.exp_buffer:
                        discounted_reward += r_i * _gamma
                        _gamma *= self.gamma
                    # We want to fill buffer only with form explorator
                    if self.agent_type == "exploration":
                        try:
                            replay_queue.put_nowait([state_0, action_0, discounted_reward, next_state, done, _gamma])
                        except:
                            pass

                state = next_state

                if done or num_steps == self.max_steps:
                    # add rest of experiences remaining in buffer
                    while len(self.exp_buffer) != 0:
                        state_0, action_0, reward_0 = self.exp_buffer.popleft()
                        discounted_reward = reward_0
                        _gamma = self.gamma
                        for (_, _, r_i) in self.exp_buffer:
                            discounted_reward += r_i * _gamma
                            _gamma *= self.gamma
                        if self.agent_type == "exploration":
                            try:
                                replay_queue.put_nowait(
                                    [state_0, action_0, discounted_reward, next_state, done, _gamma])
                            except:
                                pass
                    break

                num_steps += 1


            # Saving agent
            reward_outperformed = episode_reward - best_reward > 1
            time_to_save = self.local_episode % self.num_episode_save == 0
            if self.n_agent == 0 and (time_to_save or reward_outperformed):
                if episode_reward > best_reward:
                    best_reward = episode_reward
                # self.save(f"local_episode_{self.local_episode}_reward_{best_reward:4f}")

            rewards.append(episode_reward)
            if self.agent_type == "exploration" and self.local_episode % 1 == 0:
                self.update_actor_learner(learner_w_queue, training_on)

        empty_torch_queue(replay_queue)
        print(f"Agent {self.n_agent} done.")

'''
    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()
        self.target_actor.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
        self.target_actor.load_checkpoint()
'''