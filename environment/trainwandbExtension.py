import sys
from stable_baselines3 import PPO
from handover_env import HandoverEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback
import os


def main(model_no):
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 50100000,  # You can adjust this to how many more steps you want to train
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
        return Monitor(HandoverEnv(render_mode="rgb_array",tasks_to_complete = ["object_move_place", "object_move_handover", "object_move_lift", "panda_giver_retreat", "panda_giver_grasp","panda_reciever_to_giver","panda_reciever_grasp"], max_episode_steps = 400))

    env = DummyVecEnv([make_env] * 4)

    env = VecVideoRecorder(
        env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % 500000 == 0,
        video_length=600,
    )

    # Load the pre-trained model
    model_path = f"/home-mscluster/abrickhill/research/RR/environment/models/PPO/pickupFurther{model_no}.zip"
    if os.path.exists(model_path):
        model = PPO.load(model_path, env=env)
        print(f"Loaded pre-trained model from {model_path}")
    else:
        print(f"No pre-trained model found at {model_path}. Training from scratch.")
        model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")

    # Continue training
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
        reset_num_timesteps=False  # This ensures that the timestep count continues from where it left off
    )

    run.finish()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <iteration_number>")
        sys.exit(1)
    
    try:
        iteration = int(sys.argv[1])
    except ValueError:
        print("Error: Iteration number must be an integer")
        sys.exit(1)
    
    main(iteration)