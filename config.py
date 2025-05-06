# class Config:
#     ITERATION = 3
#     MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
    
#     def __init__(self) -> None:
#         from dotenv import dotenv_values
#         from helpers.utils import get_absolute_path
#         tokens_dict = dotenv_values(get_absolute_path(".env.development"))

#         print("tokens_dict")
#         print(tokens_dict)

#         self.HF_TOKEN = tokens_dict["HF_TOKEN"]
#         self.OPENAI_TOKEN = tokens_dict["OPENAI_TOKEN"]
#         self.PERSPECTIVE_API_TOKEN = tokens_dict["PERSPECTIVE_API_TOKEN"]

    
# CONFIG = Config()


class Config:
    ITERATION = 3
    MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
    
    def __init__(self) -> None:
        from dotenv import dotenv_values
        from helpers.utils import get_absolute_path
        tokens_dict = dotenv_values(get_absolute_path(".env.development"))

        # Handle missing keys gracefully
        self.HF_TOKEN = tokens_dict.get("HF_TOKEN")
        self.OPENAI_TOKEN = tokens_dict.get("OPENAI_TOKEN")
        self.PERSPECTIVE_API_TOKEN = tokens_dict.get("PERSPECTIVE_API_TOKEN")



CONFIG = Config()
