from envs.flood.flood import FloodController
from envs.flood.floodagent_controller import FloodAgentController

from gym.envs.registration import register

register(
    id="flood-v0",
    entry_point="envs.flood.flood_gym:FloodEnv"
)