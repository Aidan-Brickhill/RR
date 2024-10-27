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
episodes_per_model = 500

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
    model_path = f"environment/models/PPO_Final/BEST_MODEL_PER_PF/pf_{penalty_factor}"
    print(f"Running Penalty Factor {penalty_factor} ==========================")
    # Set the model number
    model_success_rate = 0
    model_violations = 0
    
    # Load the model
    try:
        model = PPO.load(model_path, env=env)
    except FileNotFoundError:
        print(f"Model {model_path} not found. Skipping to next model.")
    
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

    # Store the success rate for this model
    success_rate = model_success_rate / episodes_per_model * 100
    violation = model_violations / episodes_per_model

    # Store the results for the penalty factor
    penalty_factors_success_rates_violations[penalty_factor] = {
        'success_rate': success_rate,
        'violation': violation,
    }

    print(f"\nSuccess Rate for Penalty Factor {penalty_factor}: {success_rate:.2f}%")
    print(f"Violations for Penalty Factor {penalty_factor}: {violation:.2f}")

env.close()

# Prepare data for plotting
penalty_factors = list(penalty_factors_success_rates_violations.keys())
success_rates = [data['success_rate'] for data in penalty_factors_success_rates_violations.values()]
violations = [data['violation'] for data in penalty_factors_success_rates_violations.values()]

# Save the data to a CSV
csv_filename = 'best_penalty_factors_results.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['Penalty Factor', 'Success Rates', 'Violations']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for i in range(len(penalty_factors)):
        writer.writerow({
            'Penalty Factor': penalty_factors[i],
            'Success Rates': success_rates[i],
            'Violations': violations[i]
        })

# Plotting success rates
plt.figure(figsize=(8, 6))

# Make sure std dev doesn't go below 0


bars = plt.bar(penalty_factors, success_rates, color='skyblue', capsize=5)
plt.xlabel('Penalty Factor')
plt.ylabel('Mean Success Rate (%)')
plt.title('Mean Success Rates by Penalty Factor')
plt.xticks(rotation=45)

# Adding value labels on bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.1f}', va='bottom')  # va: vertical alignment

plt.tight_layout()
plt.savefig('best_success_rates.png')

# Plotting safety violations in a separate figure
plt.figure(figsize=(8, 6))

bars = plt.bar(penalty_factors, violations, color='salmon', capsize=5)
plt.xlabel('Penalty Factor')
plt.ylabel('Mean Violations')
plt.title('Mean Safety Violations by Penalty Factor')
plt.xticks(rotation=45)

# Adding value labels on bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.1f}', va='bottom')  # va: vertical alignment

plt.tight_layout()
plt.savefig('best_safety_violations.png')
