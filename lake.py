import gym

env = gym.make('FrozenLake-v0')

print(env.observation_space)
print(env.action_space)

# Not slippery

from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
)
env = gym.make('FrozenLakeNotSlippery-v0')

env.reset()

# Move right
# env.step(2)
# env.render()


# Random actions

# env.reset()
# done = False
# while not done:
#     env.render()
#     action = env.action_space.sample()
#     _, _, done, _ = env.step(action)

# Q-learning

import numpy as np

Q = np.zeros([env.observation_space.n, env.action_space.n])

# Learning parameters
alpha = 0.1
gamma = 0.95
num_episodes = 2000
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

        # if i == 1999:
        #     print(i)

    rs[i] = r_sum_i

## Plot reward vs episodes

import matplotlib.pyplot as plt
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


