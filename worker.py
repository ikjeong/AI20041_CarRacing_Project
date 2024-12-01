import gymnasium as gym
import torch
import torch
from itertools import count
from environment import CarEnvironment
import torch.multiprocessing as mp

def worker(agent, episode_duration, rewards_per_episode, average_episode_loss, statistics_lock):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('CarRacing-v2', continuous=False)
    env = CarEnvironment(env)

    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    episode_total_reward = 0
    
    for t in count():
        action = agent.select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        episode_total_reward += reward
        done = terminated or truncated

        if terminated:
            next_state = None
            print("Finished the lap successfully!")
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        with agent.lock_memory:
            agent._shared_memory.put((state, action, next_state, reward))
        state = next_state

        if done:
            ll = agent.get_loss()
            if len(ll) > 0:
                with statistics_lock:
                    episode_duration.append(t+1)
                    rewards_per_episode.append(episode_total_reward)
                    average_episode_loss.append(sum(ll) / len(ll))
            break
