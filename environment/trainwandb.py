from stable_baselines3 import PPO
from handover_env import HandoverEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 50100000,
    "env_name": "HandoverEnv",
}

run = wandb.init(
    project="Safe_Robot_Handover",
    config=config,
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True,    
)

def make_env():
    return Monitor(HandoverEnv(render_mode="rgb_array",tasks_to_complete = ["object_move_place", "object_move_handover", "object_move_lift", "panda_giver_retreat", "panda_giver_fetch","panda_reciever_to_giver","panda_reciever_fetch"], max_episode_steps = 400))

env= DummyVecEnv([make_env] * 4)

env = VecVideoRecorder(
    env,
    f"videos/{run.id}",
    record_video_trigger=lambda x: x % 500000 == 0,
    video_length=600,
)

model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)

run.finish()