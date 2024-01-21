import sys
import os
PATH = os.path.dirname(os.path.abspath(__file__))
# go to parent directory until the folder name is HAZARD
while os.path.basename(PATH) != "HAZARD":
    PATH = os.path.dirname(PATH)
sys.path.append(PATH)
sys.path.append(os.path.join(PATH, "ppo"))

import torch

model_dir = {
    "fire": os.path.join(PATH, "ppo/trained_models/ppo/fire-v0.pt"),
    "flood": os.path.join(PATH, "ppo/trained_models/ppo/flood-v0.pt"),
    "wind": os.path.join(PATH, "ppo/trained_models/ppo/wind-v0.pt"),
}

class RLAgent:
    def __init__(self, task):
        self.task = task
        if task == "wind":
            self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        elif task == "fire":
            self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.model:torch.nn.Module = torch.load(model_dir[task])[0].to(self.device)
        self.model.eval()
        self.agent_type = "rl"
        
        self.eval_hidden_state = torch.zeros((1, self.model.recurrent_hidden_state_size), device=self.device, dtype=torch.float32)
        self.eval_masks = torch.zeros(1, device=self.device, dtype=torch.float32)
        self.extras = torch.zeros((1, 8), device=self.device, dtype=torch.float32)
    
    def reset(self, goal_objects, objects_info):
        self.eval_hidden_state = torch.zeros((1, self.model.recurrent_hidden_state_size), device=self.device, dtype=torch.float32)
        self.eval_masks = torch.zeros(1, device=self.device, dtype=torch.float32)
        self.extras = torch.zeros((1, 8), device=self.device, dtype=torch.float32)
    
    def choose_target(self, state, processed_input):
        self.current_state = state
        with torch.no_grad():
            obs = torch.tensor(self.current_state["RL"], device=self.device, dtype=torch.float32).unsqueeze(0)
            # print(obs.shape, self.eval_hidden_state.shape, self.eval_masks.shape, self.extras.shape)
            _, action, _, self.eval_hidden_state = self.model.act(
                obs,
                self.eval_hidden_state,
                self.eval_masks,
                self.extras,
                deterministic=False)
        return action.item()

if __name__ == "__main__":
    agent = RLAgent("fire")
    print(agent.model)