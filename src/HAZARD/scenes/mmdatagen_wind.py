from envs.wind import WindAgentController
from envs.wind.wind_gym import *
from tdw.add_ons.object_manager import ObjectManager
from tdw.add_ons.image_capture import ImageCapture
from tdw.output_data import OutputData, SegmentationColors
import os

## this is only available for non-wind scenes
DATA_PATH = "data/room_setup_wind"
IMAGE_PATH = "../images2/wind"

def generate_images(config):
    os.system(f"rm -rf {os.path.join(IMAGE_PATH, config)}")
    os.makedirs(os.path.join(IMAGE_PATH, config))
    setup = SceneSetup(data_dir=os.path.join(DATA_PATH, config))
    controller = WindAgentController(screen_size=1024, port=12138)
    controller.init_scene(setup)

    om = ObjectManager()

    controller.add_ons.extend([om])

    controller.communicate([])
    print(om.categories)

    segmentation_colors = dict()
    backward_segmentation_colors = dict()
    cat_map = dict()
    cnt = 0
    for idx in om.objects_static:
        segm = om.objects_static[idx].segmentation_color
        segmentation_colors[idx] = segm
        cat = om.objects_static[idx].category
        if not cat in cat_map:
            cat_map[cat] = cnt
            cnt += 1
        backward_segmentation_colors[tuple(segm.astype(int).tolist())] = cat_map[cat]
    import pickle
    with open(os.path.join(IMAGE_PATH, config, "segm.txt"), "w") as f:
        print(backward_segmentation_colors, cat_map, file=f)

    positions = []
    agent_pos = setup.agent_positions[0]
    # random positions
    for i in range(20):
        for j in range(20):
            positions.append([i * 0.5 - 5 + agent_pos[0], j * 0.5 - 5 + agent_pos[2]])
    
    
    camera = ThirdPersonCamera(avatar_id="a")
    ic = ImageCapture(path=os.path.join(IMAGE_PATH, config, "images"), avatar_ids=["a"], pass_masks=["_img", "_id"])
    controller.add_ons.extend([camera, ic])
    # controller.agents[0].turn_by(23)
    # controller.communicate([])
    # print(controller.agents[0].image_frequency)
    # print(controller.add_ons)
    # for i in range(10):
    #     controller.communicate([])
    #     print(controller.agents[0].dynamic.images.keys())
    # agent = controller.agents[0]
    import random
    import tqdm
    for i in tqdm.tqdm(range(200)):
        pos = random.choice(positions)
        pos = np.array([pos[0], 1.2, pos[1]])
        rot = np.random.uniform(-180, 180)
        look = pos + np.array([np.cos(rot) * 4, 0.0, np.sin(rot) * 4])

        camera.teleport(position=TDWUtils.array_to_vector3(pos))
        camera.look_at(target=TDWUtils.array_to_vector3(look))

        controller.communicate([])
    
    controller.communicate([{"$type": "terminate"}])
    controller.socket.close()


if __name__ == "__main__":
    os.makedirs(IMAGE_PATH, exist_ok=True)
    for config in os.listdir(DATA_PATH):
        if not os.path.exists(os.path.join(DATA_PATH, config, "log.txt")):
            continue
        print(config)
        generate_images(config)