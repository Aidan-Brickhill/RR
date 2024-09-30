import wandb
import math
import os
import numpy as np
from stable_baselines3 import PPO
from handover_env import HandoverEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback

class EpisodeViolationsCallBack(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_violations = []
        self.current_episode_violations = 0

    def _on_step(self) -> bool:
        # Get the current episode violations
        # current_violations = self.training_env.unwrapped.episode_violations
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
            # wandb.log({"global_step": self.num_timesteps, "rollout/ep_all_safety_violation_mean": mean_violations})
            wandb.log({"rollout/ep_all_safety_violation_mean": mean_violations}, step=self.num_timesteps)

        return True
    
class RobotToRobotViolationsCallBack(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.robot_to_robot_violations = []
        self.current_robot_to_robot_violations = 0

    def _on_step(self) -> bool:
        # Get the current episode violations
        # current_violations = self.training_env.unwrapped.robot_to_robot_violations
        current_violations = self.training_env.get_attr("robot_to_robot_violations")[0]
        
        # If the episode has ended (current violations is 0), log the previous episode's violations
        if current_violations == 0 and self.current_robot_to_robot_violations > 0:
            self.robot_to_robot_violations.append(self.current_robot_to_robot_violations)
            self.current_robot_to_robot_violations = 0
        else:
            self.current_robot_to_robot_violations = current_violations

        # Calculate and log the mean violations
        if len(self.robot_to_robot_violations) > 0:
            mean_violations = np.mean(self.robot_to_robot_violations)
            # wandb.log({"global_step": self.num_timesteps, "rollout/ep_robot_to_robot_safety_violation_mean": mean_violations})
            wandb.log({"rollout/ep_robot_to_robot_safety_violation_mean": mean_violations}, step=self.num_timesteps)

        return True
    
class RobotToTableViolationsCallBack(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.robot_to_table_violations = []
        self.current_robot_to_table_violations = 0

    def _on_step(self) -> bool:
        # Get the current episode violations
        # current_violations = self.training_env.unwrapped.robot_to_table_violations
        current_violations = self.training_env.get_attr("robot_to_table_violations")[0]
        
        # If the episode has ended (current violations is 0), log the previous episode's violations
        if current_violations == 0 and self.current_robot_to_table_violations > 0:
            self.robot_to_table_violations.append(self.current_robot_to_table_violations)
            self.current_robot_to_table_violations = 0
        else:
            self.current_robot_to_table_violations = current_violations

        # Calculate and log the mean violations
        if len(self.robot_to_table_violations) > 0:
            mean_violations = np.mean(self.robot_to_table_violations)
            # wandb.log({"global_step": self.num_timesteps, "rollout/ep_robot_to_table_safety_violation_mean": mean_violations})
            wandb.log({"rollout/ep_robot_to_table_safety_violation_mean": mean_violations}, step=self.num_timesteps)

        return True
    
class RobotToObjectViolationsCallBack(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.robot_to_object_violations = []
        self.current_robot_to_object_violations = 0

    def _on_step(self) -> bool:
        # Get the current episode violations
        # current_violations = self.training_env.unwrapped.robot_to_object_violations
        current_violations = self.training_env.get_attr("robot_to_object_violations")[0]
        
        # If the episode has ended (current violations is 0), log the previous episode's violations
        if current_violations == 0 and self.current_robot_to_object_violations > 0:
            self.robot_to_object_violations.append(self.current_robot_to_object_violations)
            self.current_robot_to_object_violations = 0
        else:
            self.current_robot_to_object_violations = current_violations

        # Calculate and log the mean violations
        if len(self.robot_to_object_violations) > 0:
            mean_violations = np.mean(self.robot_to_object_violations)
            # wandb.log({"global_step": self.num_timesteps, "rollout/ep_robot_to_object_violation_mean": mean_violations})
            wandb.log({"rollout/ep_robot_to_object_violation_mean": mean_violations}, step=self.num_timesteps)

        return True

class ObjectDroppedViolationsCallBack(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.object_dropped_violations = []
        self.current_object_dropped_violations = 0

    def _on_step(self) -> bool:
        # Get the current episode violations
        # current_violations = self.training_env.unwrapped.object_dropped_violations
        current_violations = self.training_env.get_attr("object_dropped_violations")[0]
        
        # If the episode has ended (current violations is 0), log the previous episode's violations
        if current_violations == 0 and self.current_object_dropped_violations > 0:
            self.object_dropped_violations.append(self.current_object_dropped_violations)
            self.current_object_dropped_violations = 0
        else:
            self.current_object_dropped_violations = current_violations

        # Calculate and log the mean violations
        if len(self.object_dropped_violations) > 0:
            mean_violations = np.mean(self.object_dropped_violations)
            # wandb.log({"global_step": self.num_timesteps, "rollout/ep_object_dropped_safety_violation_mean": mean_violations})
            wandb.log({"rollout/ep_object_dropped_safety_violation_mean": mean_violations}, step=self.num_timesteps)

        return True

class WandbModelSaver(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_name = f"model_{self.n_calls}_steps"
            path = os.path.join(self.save_path, f"{model_name}.zip")
            self.model.save(path)
            
            # Create and log a wandb Artifact
            artifact = wandb.Artifact(name=model_name, type="model")
            artifact.add_file(path)
            wandb.log_artifact(artifact)
            
            if self.verbose > 0:
                print(f"Saving model checkpoint to {path}")
        return True

config = {
    "policy_type": "MlpPolicy",
    # "total_timesteps": 50100000,
    "total_timesteps": 100000,
    "env_name": "HandoverEnv",
}

run = wandb.init(
    project="Safe_Robot_Handover",
    config=config,
    sync_tensorboard=True,
    # sync_tensorboard=False,
    monitor_gym=True,
    save_code=True,    
)

def make_env():
    return Monitor(HandoverEnv(render_mode="rgb_array",tasks_to_complete = ["object_move_place", "object_move_handover", "object_move_lift", "panda_giver_retreat", "panda_giver_grasp","panda_reciever_to_giver","panda_reciever_grasp"], max_episode_steps = 400))

env= DummyVecEnv([make_env] * 4)

env = VecVideoRecorder(
    env,
    f"videos/{run.id}",
    record_video_trigger=lambda x: x % 500000 == 0,
    video_length=600,
)

# rovbot arm contorl hyperparamters https://arxiv.org/html/2407.02503v1
# model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}", 
#             learning_rate=0.0153,
#             n_steps=559,
#             batch_size=193,
#             gamma=0.9657,
#             ent_coef=0.0548,
#             vf_coef=0.3999,
#             max_grad_norm=9.4229,
#             gae_lambda=0.8543,
#             clip_range=0.2865,
#             )

# default
model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")

# Calculate the frequency to save the model (e.g., 5 times during training)
save_freq = math.ceil(config["total_timesteps"] / 5)

# Create an instance of the custom callback
wandb_saver = WandbModelSaver(
    save_freq=save_freq,
    save_path=f"models/{run.name}",
    verbose=1
)


callbacks = [
    EpisodeViolationsCallBack(),
    RobotToRobotViolationsCallBack(),
    RobotToTableViolationsCallBack(),
    RobotToObjectViolationsCallBack(),
    ObjectDroppedViolationsCallBack(),
    wandb_saver,
    WandbCallback(
        gradient_save_freq=100,
        verbose=2,
    )
]

model.learn(
    total_timesteps=config["total_timesteps"],
    callback=callbacks,
)

run.finish()