import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data from CSV
df = pd.read_csv('penalty_factors_results.csv')
penalty_factors = ["0","1","2","5","10","20"]

# Extract necessary columns
mean_rewards = df['Mean Reward']
std_dev_rewards = df['Std Dev Reward']
mean_violations = df['Mean Violations']
std_dev_violations = df['Std Dev Violations']

# Sample sizes (assuming you know these or have them available)
# If n is different for each, adjust accordingly.
n_samples = 1000  # Replace with actual sample sizes if different

# Calculate 95% confidence intervals
ci_rewards = 1.96 * (std_dev_rewards / np.sqrt(n_samples))
ci_violations = 1.96 * (std_dev_violations / np.sqrt(n_samples))


# Plotting success rates
plt.figure(figsize=(8, 6))

# Make sure std dev doesn't go below 0

bars = plt.bar(penalty_factors, mean_rewards, yerr=ci_rewards, color='skyblue', capsize=5)
plt.xlabel('Penalty Factor')
plt.ylabel('Mean Success Rate (%)')
plt.title('Mean Success Rates by Penalty Factor')
plt.xticks(rotation=45)

# Adding value labels on bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.1f}', va='bottom')  # va: vertical alignment

plt.tight_layout()
plt.savefig('normal_success_rates.pdf')

# Plotting safety violations in a separate figure
plt.figure(figsize=(8, 6))

bars = plt.bar(penalty_factors, mean_violations, yerr=ci_violations, color='salmon', capsize=5)
plt.xlabel('Penalty Factor')
plt.ylabel('Mean Violations')
plt.title('Mean Safety Violations by Penalty Factor')
plt.xticks(rotation=45)

# Adding value labels on bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.1f}', va='bottom')  # va: vertical alignment

plt.tight_layout()
plt.savefig('normal_safety_violations.pdf')