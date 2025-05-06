from abc import ABC, abstractmethod

class RewardWrapper(ABC):
    
    @abstractmethod
    def calculate_reward_dict(self, input) -> dict:
        raise NotImplementedError(f"The reward calculation method is not implemented for {self.__class__.__name__}.")
    
    @abstractmethod
    def calculate_reward_aggregated(self, input, aggregation_policy=None) -> float:
        raise NotImplementedError(f"The reward aggregation method is not defined for {self.__class__.__name__}.")



class PerspectiveReward(RewardWrapper):
    def __init__(self) -> None:
        super().__init__()
        
        from helpers.perspective_client import PerspectiveClient
        from config import CONFIG
        self.perspective_client = PerspectiveClient(api_key=CONFIG.PERSPECTIVE_API_TOKEN)
        
        from helpers.enums import ToxicityAttributes
        self.attribute_list = [attribute.name for attribute in ToxicityAttributes]
        
    def calculate_reward_dict(self, input) -> dict:
         output_scores, truncated = self.perspective_client.analyze_text(input, self.attribute_list)
         return output_scores, truncated
     
    def calculate_reward_aggregated(self, input, aggregation_policy=None) -> float:
        scores_dict, truncated = self.calculate_reward_dict(input)
        
        if not aggregation_policy:
            score_values = [value for value in scores_dict.values() if isinstance(value, float)]
            if score_values:
                average = sum(score_values) / len(score_values)
                return average, scores_dict, truncated
            else:
                return 0, scores_dict, truncated
        
        raise NotImplementedError(f"The {aggregation_policy} policy is not implemented for {self.__class__.__name__}.")
        
            
# Demo of how to work with it:
# reward_calculator = PerspectiveReward()
# input = "A dummy sentence to test the perspective API"
# scores_dict = reward_calculator.calculate_reward_dict(input)
# scores_avg = reward_calculator.calculate_reward_aggregated(input)
# print(f"Scores Dict = {scores_dict}")
# print(f"Scores Avg = {scores_avg}")