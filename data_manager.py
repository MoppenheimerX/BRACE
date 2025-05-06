from abc import ABC, abstractmethod


class SentenceDBiasDataset(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.dataset = self.load_data()
    
    def get_data(self):
        return self.dataset
    
    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def filter_data_by_attribute_thresholds(self, attributes):
        pass

    @abstractmethod
    def get_sentence(self, data_entry):
        pass
    
    

class RealToxicityPromptDataset(SentenceDBiasDataset):
    def load_data(self):
        from datasets import load_dataset
        dataset = load_dataset("allenai/real-toxicity-prompts")['train']
        return dataset
    
    def filter_data_by_attribute_thresholds(self, attributes):
        def filter_function(x):
            for attribute_name, first_threshold, second_threshold in attributes:
                if not (x['prompt'].get(attribute_name) and x['prompt'][attribute_name] > first_threshold):
                    return False
                if not (x['continuation'].get(attribute_name) and x['continuation'][attribute_name] > second_threshold):
                    return False
            return True

        return self.dataset.filter(filter_function)

    def get_sentence(self, data_entry):
        return data_entry['prompt']['text'] + " " + data_entry['continuation']['text']
    
    def get_first_part_of_sentence(self, data_entry):
        return data_entry['prompt']['text']
    
    
# # Demo of how to work with it:
# real_toxic_ds = RealToxicityPromptDataset()
# for input_entry in real_toxic_ds.get_data():
#       first_part_of_sentence = real_toxic_ds.get_sentence(input_entry)
#       print(first_part_of_sentence)
#       break