from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from prompt_manager import ToxicityBasedPromptPool, PromptManager
from data_manager import RealToxicityPromptDataset
from reward_wrapper import PerspectiveReward
from prompt_manager import ToxicityBasedPromptPool, PromptManager
from llm_wrapper import GPTWrapper
from config import CONFIG
from helpers.enums import GPTVariants
from typing import Optional
import random

from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")



class Actions(Enum):
    TOXICITY = 0
    PROFANITY = 1
    SEXUALLY_EXPLICIT = 2
    # FLIRTATION = 3
    IDENTITY_ATTACK = 3
    THREAT = 4
    INSULT = 5
    SEVERE_TOXICITY = 6
    # NO_TOXICITY = 7
    
class GPTVariants(Enum):
    GPT_4_MINI = "gpt-4o-mini"
    GPT_3_TURBO = "gpt-3.5-turbo-0125"


class LLM_Env(gym.Env):
    # metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, test_mode=False, base_prompt=None, feedback_type=None, render_mode=None, BERT_hidden_layer_size=768, init_action=0, episode_len=10,
                num_bins=10, bins_low = 0.1, bins_high = 0.55):
        self.Action_size = len(Actions)  # The size of the square grid
        self.Encoded_txt_size = BERT_hidden_layer_size  # The size of the PyGame window
        self.test_mode = test_mode
        self.init_action = init_action
        self.action_old = init_action
        self.action = init_action
        self.n_steps = 0
        self.completion_counter = 0
        self.bins_low = bins_low
        self.bins_high = bins_high
        

        self.allowed_feedback_types = ["zero_shot", "few_shot", "no_shot"]
        assert feedback_type in self.allowed_feedback_types, f"Invalid feedback_type: {feedback_type}. Expected one of {self.allowed_feedback_types}."
        self.feedback_type = feedback_type

        self.step_counter = 0
        self.episode_len = episode_len

        # Initialize a dictionary to store trajectories
        self.trajectory = {
            "actions": [],
            "sentences": [],
            "rewards": [],
            "scores": []
        }

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        # self.observation_space = spaces.Dict(
        #     {
        #         "Input_Encoded": spaces.Box(-1000.0, 1000.0, shape=(self.Encoded_txt_size,), dtype=np.float32),
        #         "Action_Old": spaces.Box(0, self.Action_size - 1, shape=(1,), dtype=np.int32),
        #         "Response_Encoded": spaces.Box(-1000.0, 1000.0, shape=(self.Encoded_txt_size,), dtype=np.float32),
        #         "Response_Encoded_old": spaces.Box(-1000.0, 1000.0, shape=(self.Encoded_txt_size,), dtype=np.float32)
        #     }
        # )

        self.observation_space = spaces.Dict(
            {
                "Response_Encoded": spaces.Box(-1000.0, 1000.0, shape=(self.Encoded_txt_size,), dtype=np.float32)
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(self.Action_size)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self.prompt_pool = ToxicityBasedPromptPool()
        self._action_to_prompt = {
            Actions.TOXICITY.value: self.prompt_pool.get_prompt_pool()[Actions.TOXICITY.value],
            Actions.PROFANITY.value: self.prompt_pool.get_prompt_pool()[Actions.PROFANITY.value],
            Actions.SEXUALLY_EXPLICIT.value: self.prompt_pool.get_prompt_pool()[Actions.SEXUALLY_EXPLICIT.value],
            # Actions.FLIRTATION.value: self.prompt_pool.get_prompt_pool()[Actions.FLIRTATION.value],
            Actions.IDENTITY_ATTACK.value: self.prompt_pool.get_prompt_pool()[Actions.IDENTITY_ATTACK.value],
            Actions.THREAT.value: self.prompt_pool.get_prompt_pool()[Actions.THREAT.value],
            Actions.INSULT.value: self.prompt_pool.get_prompt_pool()[Actions.INSULT.value],
            Actions.SEVERE_TOXICITY.value: self.prompt_pool.get_prompt_pool()[Actions.SEVERE_TOXICITY.value],
            # Actions.NO_TOXICITY.value: self.prompt_pool.get_prompt_pool()[Actions.NO_TOXICITY.value],
        }

        self.llm = GPTWrapper(GPTVariants.GPT_3_TURBO.value, CONFIG.OPENAI_TOKEN)
        # self.dataset = RealToxicityPromptDataset()
        self.reward_model = PerspectiveReward()

        

        # if not hasattr(self, "random_indexes"):
        #     data = self.dataset.get_data()
        #     data_length = len(data)
        #     print("Length of dataset: ", data_length)
            
        #     data_portion = 0.20

        #     # Create a list of all indices in order
        #     self.all_indexes = list(range(data_length))
        #     self.all_indexes = self.all_indexes[:int(data_portion * len(self.all_indexes))]

        #     data_length_new = len(self.all_indexes)
        #     split_ratio = 0.20
            
        #     # Compute test set size (20% of the dataset)
        #     test_size = int(split_ratio * data_length_new)
            
        #     # Split: first 20% for test, remaining 80% for training
        #     self.test_indexes_org = self.all_indexes[:test_size]
        #     self.train_indexes_org = self.all_indexes[test_size:]
        #     self.len_test_set_org = len(self.test_indexes_org)
        #     self.test_indexes = self.all_indexes[:test_size]
        #     self.train_indexes = self.all_indexes[test_size:]
        #     self.len_train_set = len(self.train_indexes)
        #     self.len_test_set = len(self.test_indexes)
        #     print(f"Length of train set: {self.len_train_set}")
        #     print(f"Length of test set : {self.len_test_set}")

        #     self.dataset_idx = self.train_indexes
        #     self.dataset_len = self.len_train_set
        #     if self.test_mode:
        #         self.dataset_idx = self.test_indexes_org
        #         self.dataset_len = self.len_test_set_org
            
            
        #     # Now, shuffle only the training indexes if randomness is needed for training
        #     random.shuffle(self.dataset_idx)
            
        #     # Initialize a pointer for further processing (if needed)
        #     self.current_idx = 0

        self.current_bin = 0
        self.num_bins = num_bins
        self.bins = np.linspace(self.bins_low, self.bins_high, self.num_bins + 1)
        # Compute bin indices for the test set.
        
        self.dataset = np.load("train_filtered_data.npy", allow_pickle=True)
        if self.test_mode:
            self.dataset = np.load("test_filtered_data.npy", allow_pickle=True)
            self.test_reward = np.load("test_reward.npy", allow_pickle=True)
            # Compute bin indices for the test set.
            self.test_bin_indices = np.digitize(self.test_reward, self.bins, right=False) - 1
            

        self.dataset_len = len(self.dataset)
        self.dataset_idx = np.arange(self.dataset_len)

        # Initialize a pointer for further processing (if needed)
        self.current_idx = 0

        np.random.shuffle(self.dataset_idx)

        if base_prompt is None:
            raise ValueError(
                f"Invalid base_prompt: {base_prompt}. It must be a string contains your sentence."
            )
        self.base_prompt = base_prompt


        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.is_success = False

    def encode_txt(self, input):
        encoded_txt = tokenizer(input, return_tensors='pt')
        output = model(**encoded_txt)
        cls_token_vector = output.last_hidden_state[0][0]
        return cls_token_vector.detach().cpu().numpy()

    def _get_obs(self):
        
        encoded_input = self.encode_txt(self.input)
        formatted_input = self.llm.format_prompt(instruction=self.instruction, input=self.sentence)
        response = self.llm.call_llm(formatted_input)
        
        if self.step_counter == 1:
            self.response_prev = self.input
        else:
            self.response_prev = self.response

        self.response = response

        self.encoded_response_old = self.encoded_response
        self.encoded_response = self.encode_txt(response)
        observation = {
                    "Response_Encoded"     : self.encoded_response
        }
        
        

        # observation = {
        #         "Input_Encoded"        : encoded_input,
        #         "Action_Old"           : self.action_old,
        #         "Response_Encoded"     : encoded_response,
        #         "Response_Encoded_old" : self.encoded_response_old
        # }

        
        return observation, response

    def _get_info(self):
        
        if self.step_counter > 0:
            self.trajectory["actions"].append(self.action)

        self.trajectory["sentences"].append(self.response)
        self.trajectory["rewards"].append(self.reward)
        self.trajectory["scores"].append(self.scores)

        return {"response"       : self.response, 
                "is_success"     : self.is_success,
                "trajectory"     : self.trajectory,
                "Current index"  : self.current_idx}

    def reset(self, init_action: int = 0, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Initialize a dictionary to store trajectories
        self.trajectory = {
            "actions": [],
            "sentences": [],
            "rewards": [],
            "scores": []
        }
        # idx = self.dataset_idx[self.current_idx]
        # # self.input = str(self.dataset.get_data()[self.data_idx]["prompt"]["text"])
        # self.input = str(self.dataset.get_data()[idx]["prompt"]["text"]) + str(self.dataset.get_data()[idx]["continuation"]["text"])
        
        if self.test_mode:
            # Find all candidate indices in the test set that belong to the current bin
            self.candidate_indices = np.where(self.test_bin_indices == self.current_bin)[0]
            
            # If there are no candidates in this bin, search for any bin with available samples
            if len(self.candidate_indices) == 0:
                print(f"No sample available in bin {self.current_bin}; searching for a non-empty bin.")
                for b in range(self.num_bins):
                    self.candidate_indices = np.where(self.test_bin_indices == b)[0]
                    if len(self.candidate_indices) > 0:
                        self.current_bin = b
                        break

            # Randomly select one index from the candidate indices.
            self.selected_index = np.random.choice(self.candidate_indices)

            # Retrieve the sample and its associated reward and idx.
            data = self.dataset[self.selected_index]
            self.input = str(data["prompt"]["text"]) + str(data["continuation"]["text"])
            # Update the bin pointer to the next bin (with wrap-around)
            self.current_bin = (self.current_bin + 1) % self.num_bins

        else:
            idx = self.dataset_idx[self.current_idx]
            # self.input = str(self.dataset.get_data()[self.data_idx]["prompt"]["text"])
            self.input = str(self.dataset[idx]["prompt"]["text"]) + str(self.dataset[idx]["continuation"]["text"])
            self.current_idx += 1
            if self.current_idx == self.dataset_len-1:
                random.shuffle(self.dataset_idx)
                self.current_idx = 0
                self.completion_counter += 1

        self.action = init_action
        self.action_old = init_action
        self.terminated = False
        self.is_success = False
        self.step_counter = 0
        

        
        

        self.encoded_input = self.encode_txt(self.input)
        self.encoded_response = self.encoded_input
        self.encoded_response_old = self.encoded_input
        self.observation = {
                "Response_Encoded"     : self.encoded_input
                        }
        self.response = self.input
        self.reward, self.scores, truncated = self.reward_model.calculate_reward_aggregated(self.response)

        
        self.reward = -self.reward

        self.reward_old = self.reward
        

        self.info = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        # print("self.observation")
        # print(self.observation)

        return self.observation, self.info

    def step(self, action):
        
        # action = 3
        
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        self.step_counter += 1
        self.n_steps += 1

        self.action_old = self.action
        self.action = action

        self.instruction = self.base_prompt + " " + self._action_to_prompt[action]

        if self.feedback_type == self.allowed_feedback_types[0]:
            if self.step_counter == 1:
                self.sentence = self.input
            else:
                self.sentence = self.response
        elif self.feedback_type == self.allowed_feedback_types[1]:
            if self.step_counter == 1:
                self.sentence = self.input
            else:
                self.sentence = "The second-to-last response: " + self.response_prev + " The last response: " + self.response
        elif self.feedback_type == self.allowed_feedback_types[2]:
            self.sentence = self.input


        self.observation, self.response = self._get_obs()

        self.reward, self.scores, truncated = self.reward_model.calculate_reward_aggregated(self.response)

        self.reward = -self.reward

        # self.reward_new = -self.reward_new

        # self.reward = self.reward_old + self.reward_new

        # self.reward_old = self.reward_new

        # if self.step_counter == 1:
        #     self.reward = self.reward_init + self.reward 

        
        
        
        if self.reward >= -0.1 or self.step_counter == self.episode_len:
            self.terminated = True

        if self.reward > -0.1 and self.step_counter <= self.episode_len:
            self.is_success = True

        if self.reward < -0.1 and self.step_counter == self.episode_len:
            self.reward = -1.0

        

        
        # print("========== Execution Summary ==========")
        # print(f"Input: {self.input}")
        # print(f"Instruction: {self.instruction}")
        # print(f"Response: {self.response}")
        # print(f"N steps: {self.n_steps}")
        # print(f"Current idx (of {self.len_train_set}):  {self.current_idx}")
        # print(f"Completion counter: {self.completion_counter}")
        # print(f"Step Counter: {self.step_counter}")
        # print(f"Terminated: {self.terminated}")
        # # print(f"Complete sentence:     {self.input+self.response}")
        # print(f"Truncated: {truncated}")
        # print(f"is_success: {self.is_success}")
        # print(f"Action: {self.action}")
        # print(f"Action old: {self.action_old}")
        # print(f"Reward: {self.reward}")
        # print(f"Scores: {scores}")
        # print("=======================================")


        self.info = self._get_info()

        # print("self.observation")
        # print(self.observation)

        # if self.render_mode == "human":
        #     self._render_frame()

        return self.observation, self.reward, self.terminated, truncated, self.info
