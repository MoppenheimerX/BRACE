from abc import ABC, abstractmethod
import torch

class LLMWrapper(ABC):
    
    def __init__(self, model_name, access_token):
        super().__init__()
        self.model_name = model_name
        self.access_token = access_token
        self.model, self.tokenizer = self.initialize_model_tokenizer()
    
    @abstractmethod
    def initialize_model_tokenizer(self):
        raise NotImplementedError(f"The model initialization method is not implemented for {self.__class__.__name__}.")
    
    @abstractmethod
    def get_prompt_template(self) -> str | list:
        raise NotImplementedError(f"The prompt template is not defined for {self.__class__.__name__}.")
    
    @abstractmethod
    def format_prompt(self, instruction:str, input:str) -> str:
        raise NotImplementedError(f"The prompt formatter is not defined for {self.__class__.__name__}.")
    
    @abstractmethod
    def call_llm(self, input_text) -> str:
        raise NotImplementedError(f"The model text generation method is not defined for {self.__class__.__name__}.")



class GPTWrapper(LLMWrapper):
    
    def initialize_model_tokenizer(self):  
        from openai import OpenAI
        model = OpenAI(api_key=self.access_token)
        return model, None 
        
    def get_prompt_template(self) -> list:
        return [
            {"role": "system", "content": "{system_instruction}"},
            {"role": "user", "content": "{user_input}"}
        ]
    
    def format_prompt(self, instruction:str, input:str) -> str:
        template = self.get_prompt_template()
        formatted_input = [
            {"role": template[0]['role'], "content": template[0]['content'].format(system_instruction=instruction)},
            {"role": template[1]['role'], "content": template[1]['content'].format(user_input=input)}
        ]
        return formatted_input
    
    def call_llm(self, input_text) -> str:
        completion = self.model.chat.completions.create(
            model=self.model_name,
            messages=input_text
        )
        return completion.choices[0].message.content
    
    
class LLaMa2Wrapper(LLMWrapper):
    
    def initialize_model_tokenizer(self):
        from transformers import GenerationConfig
        self.llm_config = GenerationConfig(
            max_new_tokens=500,
            temperature=0.6,
            top_p=0.75,
            top_k=10,
            do_sample=True,
        )
        
        from helpers.utils import import_hf_model_and_tokenizer
        return import_hf_model_and_tokenizer(
            model_name=self.model_name, 
            access_token=self.access_token
        )  
        
    def get_prompt_template(self) -> str:
        return ("""<s>[INST] <<SYS>>\n{instruction}\n<</SYS>>\n\n{input} [/INST]\n""")
    
    def format_prompt(self, instruction:str, input:str) -> str:
        return self.get_prompt_template().format(instruction=instruction, input=input)
    
    def call_llm(self, input_text: str) -> str:
        input_ids = self.tokenizer(input_text, return_tensors='pt').input_ids.cuda()
        with torch.no_grad():
            results = self.model.generate(
                input_ids=input_ids,
                generation_config=self.llm_config
            )

            text_results = self.tokenizer.batch_decode(
                results.detach().cpu().numpy(), 
                skip_special_tokens=True
            )
        return text_results[0]

        

## Demo of how to work with it:
# from config import CONFIG 
# llm_wrapper = GPTWrapper("gpt-3.5-turbo-0125", CONFIG.OPENAI_TOKEN)

# input_text = llm_wrapper.format_prompt(
#     instruction="I will give you the first part of a sentence. Your task is to complete the sentence without changing the given first part.", 
#     input="I want to buy a "
# ) 
# print(f"Input Text: {input_text}")

# llm_output = llm_wrapper.call_llm(input_text=input_text)
# print(f"LLM Output: {llm_output}")
