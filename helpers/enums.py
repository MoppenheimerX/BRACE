from enum import Enum

class ToxicityAttributes(Enum):
    TOXICITY = 'toxicity'
    PROFANITY = 'profanity'
    SEXUALLY_EXPLICIT = 'sexually_explicit'
    # FLIRTATION = 'flirtation'
    IDENTITY_ATTACK = 'identity_attack'
    THREAT = 'threat'
    INSULT = 'insult'
    SEVERE_TOXICITY = 'severe_toxicity'
    
class GPTVariants(Enum):
    GPT_4_MINI = "gpt-4o-mini"
    GPT_3_TURBO = "gpt-3.5-turbo-0125"