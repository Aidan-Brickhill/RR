import gymnasium as gym
from handover_env import HandoverEnv

env = HandoverEnv(render_mode="human", tasks_to_complete = ["panda_giver_fetch", "panda_reciever_wait"], max_episode_steps = 400)
observation, info = env.reset()

# environment_observations = observation['observation']
# achieved_goal = observation['achieved_goal']
# desired_goal = observation['desired_goal']

# robot_observations = environment_observations[:42].copy()
# giver_robot = robot_observations[:21].copy()
# reciever_robot = robot_observations[21:].copy()

# object_position = environment_observations[42:49].copy()
# object_velocity = environment_observations[49:].copy()

# tasks_to_complete = info['tasks_to_complete']
# episode_task_completions = info['episode_task_completions']
# step_task_completions = info['step_task_completions']
terminated = False
while not terminated:
    action = env.action_space.sample()  # agent policy that uses the observation and info

    observation, reward, terminated, truncated, info = env.step(action)

    # environment_observations = observation['observation']
    # achieved_goal = observation['achieved_goal']
    # desired_goal = observation['desired_goal']

    # robot_observations = environment_observations[:36].copy()
    # giver_robot = robot_observations[:21].copy()
    # reciever_robot = robot_observations[21:].copy()
    
    # object_position = environment_observations[36:43].copy()
    # object_velocity = environment_observations[43:].copy()

    # tasks_to_complete = info['tasks_to_complete']
    # episode_task_completions = info['episode_task_completions']
    # step_task_completions = info['step_task_completions']

    # print()
    # print(robot_observations)

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

