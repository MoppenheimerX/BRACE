from gymnasium.envs.registration import register

register(
    id="LLM_Env-v0",
    entry_point="Env.gymnasium_env.envs:LLM_Env",
)
