from stable_baselines3 import PPO
from handover_env import HandoverEnv

model_path = "RR/environment/models/PPO/drop"
# model_path = "environment/models/PPO/drop"


env = HandoverEnv(render_mode="human", tasks_to_complete = ["object_move_place", "object_move_handover", "object_move_lift", "panda_giver_retreat", "panda_giver_grasp","panda_reciever_to_giver","panda_reciever_grasp"], max_episode_steps=400)
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