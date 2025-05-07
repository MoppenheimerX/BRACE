from abc import ABC, abstractmethod

class PromptManager(ABC):
    
    @abstractmethod
    def get_prompt_pool(self) -> list:
        raise NotImplementedError(f"The prompt pool method is not defined for {self.__class__.__name__}.")
    
    def get_pool_size(self) -> int:
        return len(self.get_prompt_pool())
    

class ToxicityBasedPromptPool(PromptManager):
    def __init__(self) -> None:
        super().__init__()
        
        from helpers.enums import ToxicityAttributes
        self.prompt_pool_dict = {
            ToxicityAttributes.TOXICITY: "Frame the response in a positive, constructive manner.",
            ToxicityAttributes.PROFANITY: "Avoid using profanity in the response.",
            ToxicityAttributes.SEXUALLY_EXPLICIT: "Avoid using sexually explicit language in the response.",
            # ToxicityAttributes.FLIRTATION: "Avoid using flirtation in the response.",
            ToxicityAttributes.IDENTITY_ATTACK: "Avoid using identity attacks in the response.",
            ToxicityAttributes.THREAT: "Avoid using threats in the response.",
            ToxicityAttributes.INSULT: "Avoid using insults in the response.",
            ToxicityAttributes.SEVERE_TOXICITY: "Avoid using severe toxic language in the response.",
            ToxicityAttributes.NO_TOXICITY: "The sentence contains no toxic elements; maintain its original neutral or positive tone.",
        }
        
    def get_prompt_pool(self) -> list:
        return list(self.prompt_pool_dict.values())
        
    
    