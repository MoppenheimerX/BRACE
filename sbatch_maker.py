import os
from itertools import product

# Directory where sbatch scripts will be saved
root_path = './sbatch_v1/'
os.makedirs(root_path, exist_ok=True)
# Base sbatch header lines (adjust these to your cluster settings)
sbatch_lines = [
    '#!/usr/bin/env bash',
    '#SBATCH -A naiss2024-22-1645 -p alvis',
    ##'#SBATCH -N 1 --gpus-per-node=T4:1',
    '#SBATCH -N 1',
    '#SBATCH -C NOGPU',
    '#SBATCH -t 7-00:00:00',
    'ml load CUDA-Python/12.1.0-gfbf-2023a-CUDA-12.1.1 PyTorch/2.1.2-foss-2023a-CUDA-12.1.1',  # adjust module and version if needed
    'source /mimer/NOBACKUP/groups/naiss2024-22-1645/Paper_KTH_LLM/myenv/bin/activate',
    'cd /mimer/NOBACKUP/groups/naiss2024-22-1645/Paper_KTH_LLM'
]



# The Python script to run (should use argparse to receive these parameters)
python_script = 'train.py'

# Define parameter lists (adjust values or add more options as needed)
algos        = ["PPO"]              # Algorithm name
policies     = ["MultiInputPolicy"] # Policy to use
learning_rates = [0.001, 0.0001]            # Learning rate for training
n_steps_list = [30, 60]                # Number of steps
batch_sizes  = [256]                # Batch size for training
n_epochs_list= [10, 50]                 # Number of epochs per update

env_names    = ["LLM_Env-v0"]       # Environment name
init_actions = [0]                  # Initial action for the environment
episode_lens = [4]                  # Length of each episode
feedback_types = ["zero_shot", "few_shot", "no_shot"]  # Allowed feedback types

# Generate all combinations of parameters
combinations = product(
    algos, policies, learning_rates, n_steps_list, batch_sizes, n_epochs_list,
    env_names, init_actions, episode_lens, feedback_types
)

job_count = 0
for combo in combinations:
    job_count += 1
    (alg_name, policy, learning_rate, n_steps, batch_size, n_epochs,
     env_name, init_action, episode_len, feedback_type) = combo

    # Build the command with the parameters
    command = (
        f"python {python_script}"
        f" --alg_name {alg_name}"
        f" --policy {policy}"
        f" --learning_rate {learning_rate}"
        f" --n_steps {n_steps}"
        f" --batch_size {batch_size}"
        f" --n_epochs {n_epochs}"
        f" --env_name {env_name}"
        f" --init_action {init_action}"
        f" --episode_len {episode_len}"
        f" --feedback_type {feedback_type}"
    )

    # Create an sbatch script file for this job
    sbatch_filename = os.path.join(root_path, f"{job_count}")
    with open(sbatch_filename, 'w') as f:
        for line in sbatch_lines:
            f.write(line + "\n")
        f.write("\n" + command + "\n")

    print(f"Generated {sbatch_filename} with command:")
    print(command)
