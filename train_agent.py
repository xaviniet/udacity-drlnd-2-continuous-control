from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt

import torch

from collections import deque
from ddpg_agent import Agent

def ddpg( n_episodes=500, max_t=200, train_mode=True):
    env = UnityEnvironment(file_name='./1_agent/Reacher.app')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size
    env_info = env.reset(train_mode=train_mode)[brain_name]

    states = env_info.vector_observations

    agent = Agent(state_size=states.shape[1], action_size=action_size, random_seed=2)

 
    brain_name = env.brain_names[0]
    scores = []
    scores_deque = deque(maxlen=100)
    max_score = -np.Inf
    

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=train_mode)[brain_name]
        num_agents = len(env_info.agents)
#         agent.reset()
        score = 0
        states = env_info.vector_observations
#         while True:
        for t in range(max_t):
            agent.reset()
            actions = agent.act(states)
#             actions = np.clip(actions, -1,1)
            env_info = env.step(actions)[brain_name]
            next_states= env_info.vector_observations
            rewards = env_info.rewards
#             rewards = [1.0  if x > 0.0 else 0.0 for x in rewards]
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            score += np.mean(env_info.rewards)
            if np.any(dones):
                break
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score), end="")

        if i_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score), end="")
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))   
        
        if np.mean(scores_deque)>=30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break

    env.close()
    return scores


if __name__=='__main__':
    scores = ddpg(n_episodes=1000, max_t=1000)