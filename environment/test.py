from stable_baselines3 import PPO
from handover_env import HandoverEnv

model_path = "/home/aidan/00_Ubuntu_Honours/RR/RR/environment/models/PPO/pickup.zip"
# model_path = "/home/aidan/Ubuntu/RR/RR/environment/models/PPO/model_4.zip"


# env = HandoverEnv(render_mode="human", tasks_to_complete = ["panda_giver_fetch", "object_lift", "object_move_p1","panda_reciever_wait"], max_episode_steps=300)
env = HandoverEnv(render_mode="human", tasks_to_complete = ["panda_reciever_wait", "object_move_p2", "panda_reciever_fetch","panda_reciever_place","panda_giver_retreat"], max_episode_steps=300)
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