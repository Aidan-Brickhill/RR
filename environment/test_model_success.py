from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from handover_env import HandoverEnv

# Set the penalty factors
penalty_factors = ["1","2","5","10"]
# [5.75, 23.0, 1.25, 7.5]
# Set the penalty factor success rate
penalty_factors_success_rate = []

# Set the epsiodes per model
episodes_per_model = 40

# Set the number of models 
number_of_models = 10

# Set the environemnt
env = HandoverEnv(tasks_to_complete=["object_move_place", "object_move_handover", "object_move_lift", "panda_giver_retreat", "panda_giver_grasp", "panda_reciever_to_giver", "panda_reciever_grasp"], max_episode_steps=400)

# For each penalty factor
for penalty_factor in  penalty_factors:
    
    # Set the penalty factor success rate
    penalty_factor_success_rate = 0

    # Set the penalty factor path
    model_paths = f"environment/models/PPO_Final/PF_{penalty_factor}/model_"
    print(f"Running Penality Factor {penalty_factor} ==========================")

    # Set the model number
    model_number = 0

    # For 10 models TODO: change to 10 once we have 10
    while model_number < number_of_models:
        
        # Set the model success rate
        model_success_rate = 0

        # Set the model path using the penalty factor path
        model_path = f"{model_paths}{model_number}"
        print(f"Running model {model_number}")
        
        # Load the model
        try:
            model = PPO.load(model_path, env=env)
        except FileNotFoundError:
            print(f"Model {model_number} not found. Skipping to next model.")
            model_number += 1
            continue
        
        # For episodes_per_model times, run each model
        for ep in range(episodes_per_model):
            
            print(f"\rEpisode {ep}", end="")

            obs, info = env.reset()
            terminated = False

            # Continue until until the episode has completed
            while not terminated:
                action, _states = model.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)

                # If the handover has occured, add it as a succesful run
                if info['object_handed_over']:
                    model_success_rate += 1
                    break

        # Move to the next model
        model_number += 1

        # Increment the penalty factor success rate
        penalty_factor_success_rate += model_success_rate/episodes_per_model

    penalty_factors_success_rate.append(penalty_factor_success_rate/number_of_models*100)

    env.close()

print(penalty_factors_success_rate)
# Create the bar plot
plt.bar(penalty_factors, penalty_factors_success_rate, color='blue')

# Add labels and title
plt.xlabel('Penalty Factors')
plt.ylabel('Success Rate')
plt.title('Penalty Factor vs Success Rate')

# Show the plot
plt.savefig('success_rates')