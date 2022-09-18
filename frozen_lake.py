import gym
import numpy as np
import matplotlib.pyplot as plt

# Original env

env = gym.make('FrozenLake-v0')
# Initialize table with all zeros to be uniform
Q = np.zeros([env.observation_space.n, env.action_space.n])
# Learning parameters
alpha = 0.05
gamma = 0.9999
num_episodes = 3000
# array of reward for each episode
rs = np.zeros([num_episodes])
for i in range(num_episodes):
    # Set total reward and time to zero, done to False
    r_sum_i = 0
    t = 0
    done = False

    # Reset environment and get first new observation
    s = env.reset()

    while not done:
        # Choose an action by greedily (with noise) from Q table
        a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i / 10 + 1)))

        # Get new state and reward from environment
        s1, r, done, _ = env.step(a)

        # Update Q-Table with new knowledge
        Q[s, a] = (1 - alpha) * Q[s, a] + alpha * (r + gamma * np.max(Q[s1, :]))

        # Add reward to episode total
        r_sum_i += r * gamma ** t

        # Update state and time
        s = s1
        t += 1
    rs[i] = r_sum_i

## Plot reward vs episodes

# Sliding window average
r_cumsum = np.cumsum(np.insert(rs, 0, 0))
r_cumsum = (r_cumsum[50:] - r_cumsum[:-50]) / 50
# Plot
plt.plot(r_cumsum)
plt.show()
# Print number of times the goal was reached
N = len(rs) // 10
num_Gs = np.zeros(10)
for i in range(10):
    num_Gs[i] = np.sum(rs[i * N:(i + 1) * N] > 0)

print("Rewards: {0}".format(num_Gs))
