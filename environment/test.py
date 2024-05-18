import gymnasium as gym
from handover_env import HandoverEnv

env = HandoverEnv(render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
# env = HandoverEnv(render_mode="human")

# observation, info = env.reset()
# for _ in range(1000):
#     action = env.action_space.sample()  # agent policy that uses the observation and info
#     observation, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated:
#         observation, info = env.reset()

# env.close()