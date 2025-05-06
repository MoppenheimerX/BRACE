# from data_manager import RealToxicityPromptDataset
# from reward_wrapper import PerspectiveReward
# import numpy as np
# import os
# from IPython.display import clear_output



# dataset = RealToxicityPromptDataset()
# reward_model = PerspectiveReward()

# data_length = len(dataset.get_data())

# reward_hist = []
# idx = []
# for i in range(data_length):
#     # clear_output()
#     print("data idx "+str(int(i+1))+" of "+str(data_length))
#     text = str(dataset.get_data()[i]["prompt"]["text"]) + str(dataset.get_data()[i]["continuation"]["text"])
#     reward, scores, truncated = reward_model.calculate_reward_aggregated(text)

#     reward_hist.append(reward)
#     idx.append(i)


# np.save('dataset_reward.npy', np.array(reward_hist))
# np.save('dataset_idx.npy', np.array(idx))

# import numpy as np
# import matplotlib.pyplot as plt

# # Load the reward data from the .npy file
# reward_hist = np.load('dataset_reward.npy')

# weights = np.ones_like(reward_hist) / len(reward_hist) * 100

# plt.figure(figsize=(8, 6))
# plt.hist(reward_hist, bins=30, weights=weights, edgecolor='black')
# plt.title('Histogram of Rewards (Percentage)')
# plt.xlabel('Reward')
# plt.ylabel('Percentage (%)')
# plt.grid(True)

# # Save the plot as a JPG
# plt.savefig('density_of_rewards.jpg', format='jpg', dpi=300)
# plt.show()

# # Create a histogram
# plt.figure(figsize=(8, 6))
# plt.hist(reward_hist, bins=30, edgecolor='black')
# plt.title('Histogram of Rewards')
# plt.xlabel('Reward')
# plt.ylabel('Frequency')
# plt.grid(True)

# # Save the plot as a JPG
# plt.savefig('histogram_of_rewards.jpg', format='jpg', dpi=300)

# # Show the plot
# plt.show()


# # Sort the data in ascending order
# data_sorted = np.sort(reward_hist)

# cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)

# # Create the CDF plot
# plt.figure(figsize=(8, 6))
# plt.plot(data_sorted, cdf, marker='.', linestyle='none')
# plt.xlabel('Reward')
# plt.ylabel('CDF')
# plt.title('Cumulative Distribution Function (CDF) of Rewards')
# plt.grid(True)

# # Save the plot as a JPG
# plt.savefig('CDF_of_rewards.jpg', format='jpg', dpi=300)
# plt.show()


# idx = 11000
# reward, scores, truncated = reward_model.calculate_reward_aggregated(dataset.get_data()[idx]['prompt']["text"])

# # print(dataset.get_data()[0])
# reward_hist = []
# for i in range(1000):
#     reward, scores, truncated = reward_model.calculate_reward_aggregated(dataset.get_data()[idx]['prompt']["text"])
#     reward_hist.append(reward)

# r_std = np.std(reward_hist)
# r_mean = np.mean(reward_hist)
# print(f"\n Perspective API: scores: {scores}, reward: {reward}, reward mean: {r_mean}, reward std: {r_std}")
# print(f"\n", dataset.get_data()[idx]['prompt'], "\n")


import numpy as np
import matplotlib.pyplot as plt
from data_manager import RealToxicityPromptDataset

# Load your data
dataset_idx = np.load("dataset_idx.npy")         # indices for each data point
dataset_reward = np.load("dataset_reward.npy")     # reward for each data point (range 0 to 1)
dataset = RealToxicityPromptDataset()  
data = dataset.get_data()                          # the actual dataset

Threshold = 0.1
# Filter data: keep only those with reward > 0.1 (adjust threshold as needed)
mask = dataset_reward > Threshold
filtered_idx = dataset_idx[mask]
filtered_reward = dataset_reward[mask]
filtered_data = [data[int(i)] for i in filtered_idx]

print("Number of filtered data points:", np.sum(mask))

# Optionally, save the filtered data for later use
np.save("filtered_idx.npy", filtered_idx)
np.save("filtered_reward.npy", filtered_reward)
np.save("filtered_data.npy", filtered_data)

# Define bins for the reward values between 0.1 and 0.8.
# Here we use 3 bins as an example.
num_bins = 10
bins = np.linspace(Threshold, 0.55, num_bins + 1)  # e.g., edges: [0.1, 0.3667, 0.6333, 0.8]

# Categorize each filtered reward into one of the bins.
# np.digitize returns bin indices starting at 1, so subtract 1 for a 0-based index.
bin_indices = np.digitize(filtered_reward, bins, right=False) - 1

# Determine how many samples to select from each bin so that we have 5000 total.
samples_per_bin = 5000 // num_bins

# For reproducibility
np.random.seed(42)

selected_indices = []

# Loop over each bin and select the samples
for b in range(num_bins):
    # Find positions in the filtered data corresponding to the current bin.
    indices_in_bin = np.where(bin_indices == b)[0]
    
    if len(indices_in_bin) < samples_per_bin:
        print(f"Warning: Bin {b} has only {len(indices_in_bin)} samples; expected {samples_per_bin}.")
        # If a bin doesn't have enough samples, take all available samples from that bin.
        selected = indices_in_bin
    else:
        # Randomly select 'samples_per_bin' samples without replacement.
        selected = np.random.choice(indices_in_bin, samples_per_bin, replace=False)
    
    selected_indices.append(selected)

# Combine the selected indices from all bins
selected_indices = np.concatenate(selected_indices)

# Create the train arrays using the selected indices.
train_filtered_data = [filtered_data[int(i)] for i in selected_indices]
train_reward = filtered_reward[selected_indices]
train_idx = filtered_idx[selected_indices]

# Save the train data arrays
np.save("train_filtered_data.npy", train_filtered_data)
np.save("train_reward.npy", train_reward)
np.save("train_idx.npy", train_idx)

print("Train data selection complete. Train arrays saved.")

# Determine the test filtered data as the remaining indices in the filtered data.
all_filtered_indices = np.arange(len(filtered_data))
test_indices = np.setdiff1d(all_filtered_indices, selected_indices)

test_filtered_data = [filtered_data[int(i)] for i in test_indices]
test_reward = filtered_reward[test_indices]
test_idx = filtered_idx[test_indices]

# Save the test filtered data arrays
np.save("test_filtered_data.npy", test_filtered_data)
np.save("test_reward.npy", test_reward)
np.save("test_idx.npy", test_idx)

print("Test filtered data saved.")

# Plot the histogram of the reward distribution in the train filtered data.
plt.figure(figsize=(8, 6))
plt.hist(train_reward, bins=bins, edgecolor='black')
plt.xlabel("Reward")
plt.ylabel("Number of Samples")
plt.title("Histogram of Reward Distribution for Train Filtered Data")
plt.savefig('density_of_rewards(Train_Filtered_Dataset).jpg', format='jpg', dpi=300)
plt.show()

# Plot the histogram of the reward distribution in the test filtered data.
plt.figure(figsize=(8, 6))
plt.hist(test_reward, bins=bins, edgecolor='black')
plt.xlabel("Reward")
plt.ylabel("Number of Samples")
plt.title("Histogram of Reward Distribution for Test Filtered Data")
plt.savefig('density_of_rewards(Test_Filtered_Dataset).jpg', format='jpg', dpi=300)
plt.show()





