from stable_baselines3 import PPO
from handover_env import HandoverEnv

models_dir = "models/PPO"

model_path = f"{models_dir}/100000.zip"

env = HandoverEnv(render_mode="human", tasks_to_complete=["panda_giver_fetch", "panda_reciever_fetch"], max_episode_steps=80)
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