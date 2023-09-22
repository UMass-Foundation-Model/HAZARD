"""
Example commands:
python UI.py --env wind --mode edit
python UI.py --env wind --mode edit --scene_name suburb_scene_2023

Replay mode is to navigate and possibly record an ongoing scene without modifying anything

Currently edit mode does not support adding fire or replicant.
To really try the simulation effect on editing mode, you should modify the run_additional_edit function
"""

from envs.fire.fire_gym import FireEnv
from envs.wind.wind_gym import WindEnv
from tdw.add_ons.first_person_avatar import FirstPersonAvatar
from tdw.add_ons.keyboard import Keyboard
from typing import Union
from tdw.controller import Controller
from tdw.librarian import ModelLibrarian
from tdw.tdw_utils import TDWUtils
import tkinter as tk
import os

REPLAY_PROMPT=\
"""
WASD/arrow keys: move the avatar
Mouse: look around
Left click: select an object to see its ID
Right click: exit
Space: toggle env simulation
"""

EDIT_PROMPT=\
"""
WASD/arrow keys: move the avatar
Mouse: look around
Left click: select an object
Right click: exit
Space: toggle env simulation
Return: add an object (it will open up a new window to enter its name,
        but you need to switch manually because TDW has automatic focus)
Backspace: remove an object / visual effect
"""

class UI:
    def __init__(self, env: Union[FireEnv, WindEnv]) -> None:
        self.env:Union[FireEnv, WindEnv] = env
        self.mode: str = None
        self.done: bool = False
        self.object_commands = dict()
    
    def prep_lib(self, library="models_core.json"):
        self.library = library
        self.model_names = []
        librarian = ModelLibrarian(library)
        for record in librarian.records:
            self.model_names.append(record.name)
    
    def match_nearest(self, name: str):
        import difflib
        a = difflib.get_close_matches(name, self.model_names, n=1, cutoff=0.1)
        if len(a) == 0:
            # really no matching, wtf did you enter?
            return None
        else:
            return a[0]
    
    def start_simulation(self):
        print("start simulation")
        self.env.controller.initialized = True
    
    def stop_simulation(self):
        print("stop simulation")
        self.env.controller.initialized = False
    
    def toggle_simulation(self):
        if self.env.controller.initialized:
            self.stop_simulation()
        else:
            self.start_simulation()
    
    def add_object(self):
        # show prompt
        window = tk.Tk()
        prompt = tk.Label(text="Enter the name of the model.")
        prompt.pack()

        entry = tk.Entry(window)
        entry.pack()
        entry.bind()

        self.model_name = None

        def submit():
            self.model_name = entry.get()
            window.destroy()
        
        entry.bind("<Return>", lambda event: submit())
        window.focus_force()
        window.mainloop()

        self.model_name = self.match_nearest(self.model_name)
        if self.model_name is None:
            print("No matching model found.")
            return
        idx = self.env.controller.get_unique_id()
        position = self.avatar.world_position
        print("Adding model: ", self.model_name, " with id: ", idx, "position: ", position)
        cmd = Controller.get_add_object(model_name=self.model_name,
                                        position=TDWUtils.array_to_vector3(position),
                                        object_id=idx,
                                        library=self.library)
        self.env.controller.communicate(cmd)
        self.object_commands[idx] = cmd

    def remove_object(self):
        if not self.avatar.mouse_is_over_object:
            return
        idx = self.avatar.mouse_over_object_id
        print("Removing model: ", idx)
        self.env.controller.communicate({"$type": "destroy_object", "id": idx})
        del self.object_commands[idx]
    
    def show_replay_prompts(self):
        print(REPLAY_PROMPT)

    def show_edit_prompts(self):
        print(EDIT_PROMPT)

    def set_replay(self, data_dir=None):
        self.mode = "replay"
        self.show_replay_prompts()
        self.env.reset(data_dir=data_dir)
        self.env.controller.initialized = False

        self.avatar = FirstPersonAvatar(move_speed=5.0)
        self.keyboard = Keyboard()
        self.env.controller.add_ons.append(self.avatar)
        self.env.controller.add_ons.append(self.keyboard)

        self.keyboard.listen(key="Space", function=self.toggle_simulation)
    
    def run_replay(self):
        self.done = False
        while not self.done:
            self.env.controller.communicate([])
            if self.avatar.left_button_pressed:
                print("mouse position= ", self.avatar.world_position)
                if self.avatar.mouse_is_over_object and self.avatar.left_button_pressed:
                    print("object_id= ", self.avatar.mouse_over_object_id)
            if self.avatar.right_button_pressed:
                self.done = True
        self.env.controller.communicate({"$type": "terminate"})
        self.env.controller.socket.close()

    def set_edit(self, scene_name=None):
        self.mode = "edit"
        self.show_edit_prompts()

        commands = []
        if scene_name is None:
            commands.append(TDWUtils.create_empty_room(12, 12))
        else:
            commands.append(Controller.get_add_scene(scene_name=scene_name))
        commands.append({"$type": "set_screen_size", "width": 1024, "height": 1024})
        self.env.controller.communicate(commands)
        
        self.avatar = FirstPersonAvatar(move_speed=5.0)
        self.keyboard = Keyboard()
        self.env.controller.add_ons.append(self.avatar)
        self.env.controller.add_ons.append(self.keyboard)

        self.keyboard.listen(key="Return", function=self.add_object)
        self.keyboard.listen(key="Backspace", function=self.remove_object)
        self.keyboard.listen(key="Space", function=self.toggle_simulation)

    def run_edit(self):
        self.done = False
        while not self.done:
            self.env.controller.communicate([])
            if self.avatar.left_button_pressed:
                print("mouse position= ", self.avatar.world_position)
                if self.avatar.mouse_is_over_object and self.avatar.left_button_pressed:
                    print("object_id= ", self.avatar.mouse_over_object_id)
            if self.avatar.right_button_pressed:
                self.done = True
        self.env.controller.communicate({"$type": "terminate"})
        self.env.controller.socket.close()
    
    def run_additional_replay(self):
        """
        put additional commands here, including: replicant, setting fire/wind, image capture, etc.
        """
        pass

    def run_additional_edit(self):
        """
        put additional commands here, including: replicant, setting fire/wind, image capture, etc.
        """
        pass

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="edit", choices=["replay", "edit"])

    parser.add_argument("--env", type=str, default="fire", choices=["fire", "wind"])

    """effective only when mode is replay"""
    parser.add_argument("--data_dir", type=str, default=None)

    """effective only when mode is edit"""
    parser.add_argument("--scene_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    
    parser.add_argument("--port", type=int, default=12138)
    args = parser.parse_args()

    env = None
    if args.env == "fire":
        env = FireEnv(port=args.port, launch_build=True, screen_size=512)
    elif args.env == "wind":
        env = WindEnv(port=args.port, launch_build=True, screen_size=512)

    ui = UI(env)
    ui.prep_lib("models_core.json")
    if args.mode == "replay":
        ui.set_replay(data_dir=args.data_dir)
        ui.run_additional_replay()
        ui.run_replay()
    elif args.mode == "edit":
        ui.set_edit(scene_name=args.scene_name)
        ui.run_additional_edit()
        ui.run_edit()
        if args.save_dir is not None:
            import time
            import json
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(args.save_dir, timestamp + ".json")
            with open(filename, "w") as f:
                json.dump(list(ui.object_commands.values()), f)