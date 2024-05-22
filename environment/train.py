from stable_baselines3 import PPO
from handover_env import HandoverEnv
import os
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

models_dir = "models/PPO"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

def make_env():
    return Monitor(HandoverEnv(tasks_to_complete=["panda_giver_fetch", "panda_reciever_fetch"], max_episode_steps = 80))

vec_env= DummyVecEnv([make_env] * 4)
# env = HandoverEnv(tasks_to_complete=["panda_giver_fetch", "panda_reciever_fetch"])
# env.reset()

model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=log_dir)

TIMESTEPS = 10000
for i in range(1,15):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")

vec_env.close()

# if __name__ == '__main__':
#     from stable_baselines3 import PPO
#     from handover_env import HandoverEnv
#     import os
#     from stable_baselines3.common.vec_env import SubprocVecEnv
#     from stable_baselines3.common.monitor import Monitor

#     models_dir = "models/PPO"
#     log_dir = "logs"

#     if not os.path.exists(models_dir):
#         os.makedirs(models_dir)

#     if not os.path.exists(log_dir):
#         os.makedirs(log_dir)

#     def make_env():
#         def _init():
#             env = HandoverEnv(tasks_to_complete=["panda_giver_fetch", "panda_reciever_fetch"], max_episode_steps = 50)
#             return Monitor(env)
#         return _init

#     # Create multiple environments for parallel training
#     num_envs = 4
#     vec_env = SubprocVecEnv([make_env() for _ in range(num_envs)])

#     model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=log_dir)

#     TIMESTEPS = 10000
#     for i in range(1, 50):
#         model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
#         model.save(f"{models_dir}/{TIMESTEPS * i}")

#     vec_env.close()
