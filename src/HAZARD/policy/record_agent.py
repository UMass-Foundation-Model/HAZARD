
class RecordAgent:
    def __init__(self, task):
        self.agent_type = "record"
        self.agent_name = "Bob"
        self.task = task
        self.counter = 0
        assert task in ['fire', 'flood', 'wind']

    def reset(self, goal_objects, objects_info):
        pass

    def choose_target(self, state, processed_input):
        # if self.counter == 0:
        #     self.counter += 1
        #     return ("low_level.turn_by", {"angle": 90})
        self.counter += 1
        if self.counter % 2 == 0:
            return ("low_level.move_by", {"distance": 3.5})
        return ("low_level.turn_by", {"angle": 180})
