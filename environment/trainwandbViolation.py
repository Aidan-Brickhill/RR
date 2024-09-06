import numpy as np
from stable_baselines3 import PPO
from handover_env import HandoverEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback

# class CustomWandbCallback(BaseCallback):
#     def __init__(self, verbose=0):
#         super().__init__(verbose)

#     def _on_step(self) -> bool:
#         episode_violations = self.training_env.get_attr("episode_violations")[0]
#         wandb.log({"number_violations": episode_violations}, step=self.num_timesteps)
#         return True
class CustomWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_violations = []
        self.current_episode_violations = 0

    def _on_step(self) -> bool:
        # Get the current episode violations
        current_violations = self.training_env.get_attr("episode_violations")[0]
        
        # If the episode has ended (current violations is 0), log the previous episode's violations
        if current_violations == 0 and self.current_episode_violations > 0:
            self.episode_violations.append(self.current_episode_violations)
            self.current_episode_violations = 0
        else:
            self.current_episode_violations = current_violations

        # Calculate and log the mean violations
        if len(self.episode_violations) > 0:
            mean_violations = np.mean(self.episode_violations)
            wandb.log({"rollout/ep_safety_violation_mean": mean_violations}, step=self.num_timesteps)

        return True

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
    return Monitor(HandoverEnv(render_mode="rgb_array", tasks_to_complete=["object_move_place", "object_move_handover", "object_move_lift", "panda_giver_retreat", "panda_giver_grasp", "panda_reciever_to_giver", "panda_reciever_grasp"], max_episode_steps=400))

env = DummyVecEnv([make_env] * 4)

env = VecVideoRecorder(
    env,
    f"videos/{run.id}",
    record_video_trigger=lambda x: x % 500000 == 0,
    video_length=600,
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