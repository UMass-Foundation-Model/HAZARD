import json
from tdw.scene_data.scene_bounds import SceneBounds
import numpy as np

def shift(bounds):
    eps = 1e-3
    y = bounds.top[1]
    z1 = bounds.back[-1]+eps
    z2 = bounds.front[-1]-eps
    x1 = bounds.left[0]+eps
    x2 = bounds.right[0]-eps
    z = z1 + np.random.random()*(z2 - z1)
    x = x1 + np.random.random()*(x2 - x1)
    return x, y, z

def belongs_to_which_room(x: float, z: float, scene_bounds: SceneBounds):
    for i, region in enumerate(scene_bounds.regions):
        if region.is_inside(x, z):
            return i
    return -1


with open("scene_configs/room_functionals.json") as f:
    room_functionals = json.load(f)


def get_total_rooms(floorplan_scene: str) -> int:
    return len(room_functionals[floorplan_scene[0]][0])
    
    
def get_room_functional_by_id(floorplan_scene: str, floorplan_layout: int, room_id: int) -> str:
    return room_functionals[floorplan_scene[0]][floorplan_layout][room_id]

def dis(p, l, thres=9):
    for index, i in enumerate(l):
        d = (p[0]-i[0])**2+(p[1]-i[1])**2+(p[2]-i[2])**2
        if d<thres:
            return index
    return -1


def BFS(grid: np.ndarray, start):
    dist = np.ones(grid.shape) * 1e9
    dist[start[0]][start[1]] = 0
    q = [start]
    while len(q) > 0:
        u = q.pop(0)
        for v in [(u[0] + 1, u[1]), (u[0] - 1, u[1]), (u[0], u[1] + 1), (u[0], u[1] - 1)]:
            if v[0] >= 0 and v[0] < len(grid) and v[1] >= 0 and v[1] < len(grid[0]):
                if dist[v[0]][v[1]] > 1e8 and grid[v[0]][v[1]] == 2:
                    dist[v[0]][v[1]] = dist[u[0]][u[1]] + 1
                    q.append(v)
    return dist

def generate_grid(occ):
    """
    grid = 0: not in the room
    grid = 1: occupied
    grid = 2: free
    """
    boundX = [np.min(occ.positions[:, :, 0]), np.max(occ.positions[:, :, 0])]
    boundZ = [np.min(occ.positions[:, :, 1]), np.max(occ.positions[:, :, 1])]
    grid_size = 0.25
    num_grid = [int((boundX[1] - boundX[0]) / grid_size) + 5, int((boundZ[1] - boundZ[0]) / grid_size) + 5]
    origin = [int(-boundX[0] / grid_size) + 2, int(-boundZ[0] / grid_size) + 2]

    grid = np.zeros(num_grid, dtype=int)
    for i in range(len(occ.occupancy_map)):
        for j in range(len(occ.occupancy_map[i])):
            if occ.occupancy_map[i][j] == 1:
                p = occ.positions[i, j]
                grid[int(p[0] / grid_size) + origin[0]][int(p[1] / grid_size) + origin[1]] = 1
            else:
                p = occ.positions[i, j]
                grid[int(p[0] / grid_size) + origin[0]][int(p[1] / grid_size) + origin[1]] = 2

    # f = open("occ.txt", "w")
    # for i in range(len(grid)):
    #     for j in range(len(grid[i])):
    #         if grid[i][j] == 1:
    #             f.write("x")
    #         else:
    #             f.write(" ")
    #     f.write("\n")
    # f.close()
    return grid, origin, grid_size

def grid_to_real(position, origin, grid_size):
    if not isinstance(position, list):
        position = position.tolist()
    return [position[0] * grid_size - origin[0] * grid_size, 0.0, position[1] * grid_size - origin[1] * grid_size]

def real_to_grid(position, origin, grid_size):
    if not isinstance(position, list):
        position = position.tolist()
    if len(position) > 2:
        position = [position[0], position[2]]
    return [int((position[0] + origin[0] * grid_size) / grid_size), int((position[1] + origin[1] * grid_size) / grid_size)]