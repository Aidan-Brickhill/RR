from stable_baselines3 import PPO
from handover_env import HandoverEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 20100000,
    "env_name": "HandoverEnv",
}

run = wandb.init(
    project="Safe_Robot_Handover",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

def make_env():
    # pickup
    # return Monitor(HandoverEnv(render_mode="rgb_array",tasks_to_complete = ["panda_giver_fetch", "object_lift", "object_move_p1","panda_reciever_wait"], max_episode_steps = 300))
    # handover
    return Monitor(HandoverEnv(render_mode="rgb_array",tasks_to_complete = ["panda_reciever_wait", "object_move_p2", "panda_reciever_fetch","panda_reciever_place","panda_giver_retreat"], max_episode_steps = 300))

env= DummyVecEnv([make_env] * 4)

env = VecVideoRecorder(
    env,
    f"videos/{run.id}",
    record_video_trigger=lambda x: x % 500000 == 0,
    video_length=600,
)

model_path = "RR/environment/models/PPO/trainHandover.zip"
model = PPO.load(model_path, env=env)

model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),

)

run.finish()