from envs.fire.fire import FireController
from envs.fire.fireagent_controller import FireAgentController

from gym.envs.registration import register

register(
    id="fire-v0",
    entry_point="envs.fire.fire_gym:FireEnv"
)