from stable_baselines3 import PPO
from handover_env import HandoverEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback

class CustomWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # Log the episode violations at the end of each rollout (episode)
        episode_violations = self.training_env.get_attr("episode_violations")[0]
        wandb.log({"number_violations": episode_violations}, step=self.num_timesteps)
        # Reset the episode violations
        self.training_env.env_method("reset_episode_violations")


config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 200,
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
    return Monitor(HandoverEnv(render_mode="rgb_array", tasks_to_complete=["object_move_place", "object_move_handover", "object_move_lift", "panda_giver_retreat", "panda_giver_grasp", "panda_reciever_to_giver", "panda_reciever_grasp"], max_episode_steps=400))

env = DummyVecEnv([make_env] * 2)

env = VecVideoRecorder(
    env,
    f"videos/{run.id}",
    record_video_trigger=lambda x: x % 1000 == 0,
    video_length=100,
)

model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")

# Combine the custom callback with WandbCallback
callbacks = [
    CustomWandbCallback(),
    WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    )
]

model.learn(
    total_timesteps=config["total_timesteps"],
    callback=callbacks,
)

run.finish()