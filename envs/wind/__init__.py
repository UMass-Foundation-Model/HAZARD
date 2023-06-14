from envs.wind.wind import WindController
from envs.wind.windagent_controller import WindAgentController

from gym.envs.registration import register

register(
    id="wind-v0",
    entry_point="envs.wind.wind_gym:WindEnv"
)