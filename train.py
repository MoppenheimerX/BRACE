import datetime

def generate_logname(alg_name, env_name, policy, learning_rate, n_steps, batch_size, n_epochs,
                     init_action, episode_len, feedback_type):
    # Create a timestamp for uniqueness
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Build a logname string incorporating both algorithm and environment settings
    logname = (
        f"{alg_name}_{env_name}_{policy}_lr{learning_rate}_ns{n_steps}_"
        f"bs{batch_size}_ep{n_epochs}_ia{init_action}_el{episode_len}_ft{feedback_type}_{timestamp}"
    )
    return logname





import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description="Training and Environment Parameters")

# Training parameters
parser.add_argument("--alg_name", type=str, default="PPO", help="Algorithm name")
parser.add_argument("--policy", type=str, default="MultiInputPolicy", help="Policy to use")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
parser.add_argument("--n_steps", type=int, default=100, help="Number of steps")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs per update")

# Environment settings
parser.add_argument("--env_name", type=str, default="LLM_Env-v0", help="Name of the environment")
parser.add_argument("--init_action", type=int, default=0, help="Initial action for the environment")
parser.add_argument("--episode_len", type=int, default=10, help="Length of each episode")

# Feedback type with allowed choices
allowed_feedback_types = ["zero_shot", "few_shot", "no_shot"]
parser.add_argument("--feedback_type", type=str, choices=allowed_feedback_types, default="zero_shot",
                    help="Type of feedback. Allowed values: " + ", ".join(allowed_feedback_types))

# Parse the arguments
args = parser.parse_args()

# Assign parsed arguments to corresponding variables
alg_name = args.alg_name
policy = args.policy
learning_rate = args.learning_rate
n_steps = args.n_steps
batch_size = args.batch_size
n_epochs = args.n_epochs

env_name = args.env_name
init_action = args.init_action
episode_len = args.episode_len
feedback_type = args.feedback_type

# Optional: print the parameters to verify
print("Training Parameters:")
print("  alg_name:", alg_name)
print("  policy:", policy)
print("  learning_rate:", learning_rate)
print("  n_steps:", n_steps)
print("  batch_size:", batch_size)
print("  n_epochs:", n_epochs)
print("\nEnvironment Settings:")
print("  env_name:", env_name)
print("  init_action:", init_action)
print("  episode_len:", episode_len)
print("  feedback_type:", feedback_type)



# Generate the dynamic logname
logname = generate_logname(alg_name, env_name, policy, learning_rate, n_steps,
                           batch_size, n_epochs, init_action, episode_len, feedback_type)

save_address = "./LLM_train_results/"

# Set up the logger using the dynamically generated logname
from stable_baselines3.common.logger import configure
new_logger = configure(save_address + logname, ["stdout", "csv", "log", "tensorboard", "json"])

# Rest of your code remains the same
import gymnasium
import Env.gymnasium_env
from stable_baselines3 import PPO

base_prompt = "I will give you a sentence. Your task is to reduce the toxicity of the sentence without changing its main content."

# Create the environment with the settings incorporated
env = gymnasium.make(env_name, base_prompt=base_prompt, feedback_type=feedback_type,
                     init_action=init_action, episode_len=episode_len)

model = PPO(policy, env, learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size, 
            n_epochs=n_epochs, gamma=0.99, gae_lambda=0.95, clip_range=0.2, clip_range_vf=None, 
            normalize_advantage=True, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, use_sde=False, 
            sde_sample_freq=-1, rollout_buffer_class=None, rollout_buffer_kwargs=None, target_kl=None, 
            stats_window_size=100, tensorboard_log=save_address, policy_kwargs=None, verbose=1, seed=None, 
            device='auto', _init_setup_model=True)

from custom_callback import CustomLoggingCallback, CustomEvalCallback
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

custom_callback = CustomLoggingCallback()


eval_env = gymnasium.make(env_name, test_mode=True, base_prompt=base_prompt, feedback_type=feedback_type,
                     init_action=init_action, episode_len=episode_len)


custom_eval_callback = CustomEvalCallback(
    eval_env,
    best_model_save_path=save_address + logname,
    verbose=1,
    log_path=save_address + logname,
    eval_freq=int(10 * n_steps),
    n_eval_episodes=100,
    deterministic=True,
    render=False,
    file_name=save_address + logname + "/eval_info.csv"
)

callback = CallbackList([custom_callback, custom_eval_callback])

model.set_logger(new_logger)
model.learn(total_timesteps=100_000, log_interval=1, callback=callback)
model.save(save_address + 'Weights_' + logname)

