import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env

# Create the environment
env = gym.make('Humanoid-v4')

# Check the environment (optional, but recommended)
check_env(env, warn=True)

# Wrap the environment
vec_env = DummyVecEnv([lambda: env])

# Define the model
model = PPO("MlpPolicy", vec_env, verbose=1)

# Train the model
model.learn(total_timesteps=5000)  # Adjust the number of timesteps as needed

# Test the model
env = gym.make('Humanoid-v4', render_mode="human")
observation, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
