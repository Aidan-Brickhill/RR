from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from handover_env import HandoverEnv
import csv


# Set the penalty factors
penalty_factors = ["0","1","2","5","10","20"]
# Set the penalty factor success rate
penalty_factors_success_rate = []

# Set the epsiodes per model
episodes_per_model = 100

# Set the number of models 
number_of_models = 10

# Set the environemnt
env = HandoverEnv(tasks_to_complete=["object_move_place", "object_move_handover", "object_move_lift", "panda_giver_retreat", "panda_giver_grasp", "panda_reciever_to_giver", "panda_reciever_grasp"], max_episode_steps=400)

# List to store success rates for each penalty factor
penalty_factors_success_rates_violations = {}

# For each penalty factor
for penalty_factor in penalty_factors:
    # Initialize a list to store success rates for this penalty factor
    success_rates = []
    violations = []


    # Set the penalty factor path
    model_paths = f"environment/models/PPO_Final/PF_{penalty_factor}/model_"
    print(f"Running Penalty Factor {penalty_factor} ==========================")

    # Set the model number
    model_number = 0

    # For number_of_models
    while model_number < number_of_models:
        
        # Set the model success rate
        model_success_rate = 0
        model_violations = 0

        # Set the model path using the penalty factor path
        model_path = f"{model_paths}{model_number}"
        print(f"\nRunning model {model_number}")
        
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

            # Continue until the episode has completed
            while not terminated:
                action, _states = model.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)

                # If the handover has occurred, add it as a successful run
                if info['object_handed_over']:
                    model_success_rate += 1
                    break

            model_violations += info['episode_violations']

        # Move to the next model
        model_number += 1

        # Store the success rate for this model
        success_rates.append(model_success_rate / episodes_per_model * 100)
        violations.append(model_violations / episodes_per_model)

    # Calculate the mean and standard deviation for this penalty factor
    mean_success_rate = np.mean(success_rates)
    std_dev_success_rate = np.std(success_rates)
    mean_violations= np.mean(violations)
    std_dev_violations = np.std(violations)

    # Store the results for the penalty factor
    penalty_factors_success_rates_violations[penalty_factor] = {
        'mean_reward': mean_success_rate,
        'std_dev_reward': std_dev_success_rate,
        'mean_violations': mean_violations,
        'std_dev_violations': std_dev_violations
    }

    print(f"\nMean Success Rate for Penalty Factor {penalty_factor}: {mean_success_rate:.2f}%")
    print(f"Standard Deviation of Success Rates for Penalty Factor {penalty_factor}: {std_dev_success_rate:.2f}%")

env.close()


# Prepare data for plotting
penalty_factors = list(penalty_factors_success_rates_violations.keys())
mean_rewards = [data['mean_reward'] for data in penalty_factors_success_rates_violations.values()]
std_dev_rewards = [data['std_dev_reward'] for data in penalty_factors_success_rates_violations.values()]

mean_violations = [data['mean_violations'] for data in penalty_factors_success_rates_violations.values()]
std_dev_violations = [data['std_dev_violations'] for data in penalty_factors_success_rates_violations.values()]

# Save the data to a CSV
csv_filename = 'normal_penalty_factors_results.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['Penalty Factor', 'Mean Reward', 'Std Dev Reward', 'Mean Violations', 'Std Dev Violations']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for i in range(len(penalty_factors)):
        writer.writerow({
            'Penalty Factor': penalty_factors[i],
            'Mean Reward': mean_rewards[i],
            'Std Dev Reward': std_dev_rewards[i],
            'Mean Violations': mean_violations[i],
            'Std Dev Violations': std_dev_violations[i]
        })

# Plotting success rates
plt.figure(figsize=(8, 6))

# Make sure std dev doesn't go below 0
reward_error_lower = np.maximum(0, np.array(mean_rewards) - np.array(std_dev_rewards))
reward_error_upper = np.array(mean_rewards) + np.array(std_dev_rewards)
reward_errors = [mean_rewards - reward_error_lower, reward_error_upper - mean_rewards]

bars = plt.bar(penalty_factors, mean_rewards, yerr=reward_errors, color='skyblue', capsize=5)
plt.xlabel('Penalty Factor')
plt.ylabel('Mean Success Rate (%)')
plt.title('Mean Success Rates by Penalty Factor')
plt.xticks(rotation=45)

# Adding value labels on bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.1f}', va='bottom')  # va: vertical alignment

plt.tight_layout()
plt.savefig('normal_success_rates.png')

# Plotting safety violations in a separate figure
plt.figure(figsize=(8, 6))

# Make sure std dev doesn't go below 0
violation_error_lower = np.maximum(0, np.array(mean_violations) - np.array(std_dev_violations))
violation_error_upper = np.array(mean_violations) + np.array(std_dev_violations)
violation_errors = [mean_violations - violation_error_lower, violation_error_upper - mean_violations]

bars = plt.bar(penalty_factors, mean_violations, yerr=violation_errors, color='salmon', capsize=5)
plt.xlabel('Penalty Factor')
plt.ylabel('Mean Violations')
plt.title('Mean Safety Violations by Penalty Factor')
plt.xticks(rotation=45)

# Adding value labels on bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.1f}', va='bottom')  # va: vertical alignment

plt.tight_layout()
plt.savefig('normal_safety_violations.png')
