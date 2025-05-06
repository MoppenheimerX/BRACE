from abc import ABC, abstractmethod
from data_manager import *
from reward_wrapper import *
from prompt_manager import *

class TaskManager(ABC):
    
    def __init__(self, dataset, reward, prompt_pool, base_prompt) -> None:
        self.set_dataset(dataset)
        self.set_reward_calculator(reward)
        self.set_prompt_pool(prompt_pool)
        self._base_prompt = base_prompt
    
    @property
    def get_dataset(self):
        return self._dataset
    
    @property
    def get_reward_calculator(self):
        return self._reward_calculator
    
    @property
    def get_prompt_pool(self):
        return self._prompt_pool
    
    @property
    def get_base_prompt(self):
        return self._base_prompt
    
    @abstractmethod
    def set_dataset(self):
        raise NotImplementedError(f"The dataset is not specificed for {self.__class__.__name__}.")
    
    @abstractmethod
    def set_reward_calculator(self):
        raise NotImplementedError(f"The reward calculator is not specificed for {self.__class__.__name__}.")
    
    @abstractmethod
    def set_prompt_pool(self):
        raise NotImplementedError(f"The prompt pool is not specificed for {self.__class__.__name__}.")
    

class SentenceCompletionTask(TaskManager):
    def __init__(
        self, 
        dataset: SentenceDBiasDataset, 
        reward: PerspectiveReward, 
        prompt_pool: ToxicityBasedPromptPool,
    ) -> None:
        base_prompt = "I will give you the first part of a sentence. Your task is to complete the sentence without changing the given first part."
        super().__init__(dataset, reward, prompt_pool, base_prompt) 
        
    def set_dataset(self, dataset):
        self._dataset = dataset
    
    def set_reward_calculator(self, reward):
        self._reward = reward
    
    def set_prompt_pool(self, prompt_pool):
        self._prompt_pool = prompt_pool
    

