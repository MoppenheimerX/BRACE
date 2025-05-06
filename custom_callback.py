import csv
from stable_baselines3.common.callbacks import BaseCallback

class CustomLoggingCallback(BaseCallback):
    def __init__(self, train_csv='train_indexes.csv', test_csv='test_indexes.csv', verbose=0):
        super(CustomLoggingCallback, self).__init__(verbose)
        # self.train_csv = train_csv
        # self.test_csv = test_csv

    def _on_training_start(self) -> None:
        # Get the environment (if vectorized, choose the first one)
        if hasattr(self.training_env, "envs"):
            env = self.training_env.envs[0]
        else:
            env = self.training_env

        # Unwrap the environment if it's wrapped (e.g., by Monitor)
        if hasattr(env, "unwrapped"):
            env = env.unwrapped

        # # Save train indexes to CSV
        # with open(self.train_csv, "w", newline="") as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow(["train_indexes"])  # CSV header
        #     for idx in env.train_indexes_org:
        #         writer.writerow([idx])

        # # Save test indexes to CSV
        # with open(self.test_csv, "w", newline="") as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow(["test_indexes"])  # CSV header
        #     for idx in env.test_indexes_org:
        #         writer.writerow([idx])

    def _on_step(self) -> bool:

        # If the environment is vectorized, get the first environment.
        if hasattr(self.training_env, "envs"):
            env = self.training_env.envs[0]
        else:
            env = self.training_env

        # Unwrap the environment if it's wrapped by a Monitor or other wrappers.
        if hasattr(env, "unwrapped"):
            env = env.unwrapped


        self.logger.record("env/n_steps", env.n_steps)
        self.logger.record("env/completion_counter", env.completion_counter)
        # self.logger.record("env/step_counter", env.step_counter)
        self.logger.record("env/current_idx", env.current_idx)



        return True



import csv
from stable_baselines3.common.callbacks import EvalCallback

class CustomEvalCallback(EvalCallback):
    def __init__(self, eval_env, eval_freq=1000, n_eval_episodes=10, 
                 file_name="eval_info.csv", **kwargs):
        super(CustomEvalCallback, self).__init__(
            eval_env, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes, **kwargs
        )
        self.file_name = file_name
        self.eval_info = []  # This list will accumulate info dicts from each eval step

    def _on_eval_step(self, locals_, globals_):
        # Capture the info dictionary from the evaluation step.
        # Note: In vectorized envs, locals_["infos"] is a list of info dictionaries.
        infos = locals_.get("infos", [])
        for info in infos:
            self.eval_info.append(info)
        # Call the parent method (if needed) so that normal EvalCallback logging still happens.
        super()._on_eval_step(locals_, globals_)

    def _on_eval_end(self):
        # Save the collected evaluation info to a CSV file.
        if self.eval_info:
            # Use the keys of the first info dictionary for the CSV header.
            headers = list(self.eval_info[0].keys())
            with open(self.file_name, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                for info in self.eval_info:
                    writer.writerow(info)
        # Optionally, you can also reset the storage list if you want new logs for each eval round.
        self.eval_info = []
        # Call parent method to continue with default eval callback behavior.
        super()._on_eval_end()
