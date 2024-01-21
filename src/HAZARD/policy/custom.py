import sys
import os

PATH = os.path.dirname(os.path.abspath(__file__))
# go to parent directory until the folder name is HAZARD
while os.path.basename(PATH) != "HAZARD":
    PATH = os.path.dirname(PATH)
sys.path.append(PATH)
sys.path.append(os.path.join(PATH, "ppo"))

class CustomAgent:
    def __init__(self, task):
        self.task = task
        self.agent_type = "custom"

    def reset(self, goal_objects, objects_info):
        pass

    def choose_target(self, state, processed_input):
        pass
