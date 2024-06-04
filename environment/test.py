from stable_baselines3 import PPO
from handover_env import HandoverEnv

model_path = "/home/aidan/00_Ubuntu_Honours/RR/RR/environment/models/PPO/model.zip"
# model_path = "/home/aidan/Ubuntu/RR/RR/environment/models/PPO/model.zip"


env = HandoverEnv(render_mode="human", tasks_to_complete = ["panda_giver_fetch", "object_lift", "object_move","panda_reciever_wait", "object_stable"], max_episode_steps=300)
env.reset()

model = PPO.load(model_path, env=env)

episodes = 10 
for ep in range(episodes):
    obs, info = env.reset()
    terminated = False

    while not terminated:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

env.close()