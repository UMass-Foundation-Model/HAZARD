import rule_based


class RecordAgent:
    def __init__(self, task):
        self.agent_type = "mcts"
        self.agent_name = "Bob"
        self.task = task
        assert task in ['fire', 'flood', 'wind']

    def reset(self, goal_objects, objects_info):
        pass

    def choose_target(self, state, processed_input):
        return ("record", None)
