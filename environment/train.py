from stable_baselines3 import PPO
from handover_env import HandoverEnv
import os
from stable_baselines3.common.vec_env import DummyVecEnv

# # Custom environment creation function
# def make_handover_env():
#     return HandoverEnv(tasks_to_complete=["panda_giver_fetch", "panda_reciever_fetch"])

# # Register the custom environment to make it compatible with make_vec_env
# gym.envs.register(
#     id='HandoverEnv-v0',
#     entry_point=make_handover_env,
# )

# # Parallel environments using the registered custom environment
# vec_env = make_vec_env('HandoverEnv-v0', n_envs=4)

# # Define the model
# model = PPO("MlpPolicy", vec_env, verbose=1)

# # Train the model
# model.learn(total_timesteps=50000, progress_bar=True)
# model.save("ppo_handover")

# # Clean up and reload the model
# del model  # Remove to demonstrate saving and loading
# model = PPO.load("ppo_handover")

# # Test the model
# env = HandoverEnv(render_mode = "human", tasks_to_complete=["panda_giver_fetch", "panda_reciever_fetch"])
# obs, info= env.reset()
# while True:
#     action, _states = model.predict(obs)
#     observation, reward, terminated, truncated, info = env.step(action)
#     if reward>0:
#         print(reward)

models_dir = "models/PPO"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

def make_env():
    return HandoverEnv(tasks_to_complete=["panda_giver_fetch", "panda_reciever_fetch"])

vec_env = DummyVecEnv([make_env] * 4)
env = HandoverEnv(tasks_to_complete=["panda_giver_fetch", "panda_reciever_fetch"])
env.reset()

model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=log_dir)

TIMESTEPS = 10000
for i in range(1,50):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    # model.save(f"{models_dir}/{TIMESTEPS*i}")

env.close()
