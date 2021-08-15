import numpy as np
import yaml
import gym
import rospy
import cv2
import uuid
from PIL import Image

from envs import registration

SIZE = 5
IM_SIZE = SIZE * 100 // 3

def elucidate_distance(pos1, pos2):
    x1, y1 = pos1.x, pos1.y
    x2, y2 = pos2.x, pos2.y
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def sim_pos_to_pixel_idx(x, y, lx, ly):
    return int(x / 0.03), ly - int(y / 0.03)

def crop_image_from_start_end(sx, sy, ex, ey, habitat_index, time, recovery, total, traj_pos):
    file = "jackal_helper/models/map_cropped_bw/" + "model%d.png" %habitat_index
    im = Image.open(file) 
    im = np.asarray(im, dtype=np.int8).T
    lx, ly = im.shape[0], im.shape[1]
    sx, sy = sim_pos_to_pixel_idx(sx, sy, lx, ly)
    ex, ey = sim_pos_to_pixel_idx(ex, ey, lx, ly)
    cx, cy = (sx + ex) // 2, (sy + ey) // 2

    """  visual the trajectory, only for debugging
    for tx, ty in traj_pos:
        x, y = sim_pos_to_pixel_idx(tx, ty, lx, ly)
        im[x, y] = 128

    for x, y in zip(np.linspace(sx, ex, 10), np.linspace(sy, ey, 10)):
        im[int(x), int(y)] = 128
    """

    ll = max(max(cx, lx - cx), max(cy, ly - cy))

    imp = np.zeros((2 * ll, 2 * ll))
    imp[ll - cx: ll - cx + lx, ll - cy: ll - cy + ly] = im

    angle = - np.arctan2(ey - sy, ex - sx) / np.pi * 180
    imp = rotate_image(imp, angle)
    ix, iy = imp.shape[0], imp.shape[1]
    imp = imp[ix //2 - IM_SIZE // 2: ix //2 + IM_SIZE // 2, iy //2 - IM_SIZE // 2: iy //2 + IM_SIZE // 2]

    hash_code = str(uuid.uuid4().hex[:8]) + "h=%d,t=%.2f,r=%d,c=%d" %(habitat_index, time, recovery, total)
    cv2.imwrite("images/%s.png" %hash_code, imp * 255)

def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result

def sample_start_goal_position(habitat_index):
    file = "jackal_helper/models/map_cropped_bw/" + "model%d.png" %habitat_index
    im = Image.open(file) 
    im = np.asarray(im, dtype=np.int8).T
    im = inflat_map(im)
    cv2.imwrite("inflated.png", im.T * 255)

    lx, ly = im.shape[0] // 10, im.shape[1] // 10
    grid = np.zeros((lx, ly))
    visitable_cell = []
    for i in range(lx):
        for j in range(ly):
            cell = im[i * 10: (i + 1) * 10, j * 10: (j + 1) * 10]
            grid[i, j] = (cell == 1).all()
            if grid[i, j]: 
                visitable_cell.append((i, j))
    
    # import pdb; pdb.set_trace()
    l = len(visitable_cell)
    start_pos = visitable_cell[np.random.choice(range(l))]
    end_pos = visitable_cell[np.random.choice(range(l))]
    while (
        np.sqrt((start_pos[0] - end_pos[0]) ** 2 + (start_pos[1] - end_pos[1]) ** 2) < IM_SIZE * 1.5 // 10  # large enough
        or dijkstra_shortest_path_point_search(grid, start_pos, end_pos) is None  # connected
    ):
        start_pos = visitable_cell[np.random.choice(range(l))]
        end_pos = visitable_cell[np.random.choice(range(l))]
    grid[start_pos] = 0.5
    grid[end_pos] = 0.5
    cv2.imwrite("inflated_grid.png", grid.T * 255)
    return (start_pos[0] * 0.3, (ly - start_pos[1]) * 0.3), (end_pos[0] * 0.3, (ly - end_pos[1]) * 0.3)

def sample_start_end_position(habitat_index):
    file = "jackal_helper/models/map_cropped_bw/" + "model%d.png" %habitat_index
    im = Image.open(file) 
    im = np.asarray(im, dtype=np.int8).T
    im = inflat_map(im)
    cv2.imwrite("inflated.png", im.T * 255)

    lx, ly = im.shape[0] // 10, im.shape[1] // 10
    grid = np.zeros((lx, ly))
    visitable_cell = []
    for i in range(lx):
        for j in range(ly):
            cell = im[i * 10: (i + 1) * 10, j * 10: (j + 1) * 10]
            grid[i, j] = (cell == 1).all()
            if grid[i, j]: 
                visitable_cell.append((i, j))
    
    # import pdb; pdb.set_trace()
    l = len(visitable_cell)
    start_pos = visitable_cell[np.random.choice(range(l))]
    end_pos = visitable_cell[np.random.choice(range(l))]
    i = 0
    while (
        IM_SIZE * 0.8 // 10 > np.sqrt((start_pos[0] - end_pos[0]) ** 2 + (start_pos[1] - end_pos[1]) ** 2) or
        np.sqrt((start_pos[0] - end_pos[0]) ** 2 + (start_pos[1] - end_pos[1]) ** 2) > IM_SIZE * 1.2 // 10  # large enough
        or dijkstra_shortest_path_point_search(grid, start_pos, end_pos) is None  # connected
        or dijkstra_shortest_path_point_search(grid, start_pos, end_pos) > IM_SIZE * 1.2 // 10
    ):
        start_pos = visitable_cell[np.random.choice(range(l))]
        end_pos = visitable_cell[np.random.choice(range(l))]
        i += 1
        if i >= 1000:
            return None, None
    grid[start_pos] = 0.5
    grid[end_pos] = 0.5
    cv2.imwrite("inflated_grid.png", grid.T * 255)
    return (start_pos[0] * 0.3, (ly - start_pos[1]) * 0.3), (end_pos[0] * 0.3, (ly - end_pos[1]) * 0.3)

def is_wall_indices(im, i, j, m, n):
    o = im[i, j]
    if o == 1:
        return False
    if i > 0 and im[i - 1, j] != o:
        return True
    if i + 1 < m and im[i + 1, j] != o:
        return True
    if j > 0 and im[i, j - 1] != o:
        return True
    if j + 1 < n and im[i, j + 1] != o:
        return True
    return False

def get_adjacent_indices(i, j, m, n, radius = 8):
    low = (max(i - radius, 0), max(j - radius, 0))
    high = (min(i + radius, m), min(j + radius, n))
    indices = [(i, j)]
    for ii in range(low[0], high[0]):
        for jj in range(low[1], high[1]):
            if np.sqrt((ii - i) ** 2 + (jj - j) ** 2) < radius:
                indices.append((ii, jj))
    return indices

def inflat_map(im, radius = 8):
    inflated_im = np.copy(im)
    lx, ly = im.shape[0], im.shape[1]
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if is_wall_indices(im, i, j, lx, ly):
                for idx in get_adjacent_indices(i, j, lx, ly, radius):
                    inflated_im[idx] = 0
    return inflated_im

def dijkstra_shortest_path_point_search(grid, agent_loc, target):
    """
    Find the shortest path to the object
    :param target:
    :param agent_loc:
    :param grid:
    :return: length of the shortest path. None if path does not exist
    """
    nodes = [agent_loc]
    unvisited_nodes = np.ones((grid.shape[0], grid.shape[1])).astype(np.bool)
    unvisited_nodes[agent_loc[0], agent_loc[1]] = False
    navigable_nodes = grid
    if not navigable_nodes[target[0], target[1]]:
        return None
    count = 0
    while len(nodes) > 0:
        count += 1
        next_nodes = []
        for n in nodes:
            for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                if 0 <= n[0] + dx < grid.shape[0] and 0 <= n[1] + dy < grid.shape[1]:
                    if (n[0] + dx) == target[0] and (n[1] + dy) == target[1]:
                        return count
                    elif unvisited_nodes[n[0] + dx, n[1] + dy]:
                        unvisited_nodes[n[0] + dx, n[1] + dy] = False
                        if navigable_nodes[n[0] + dx, n[1] + dy]:
                            next_nodes.append((n[0] + dx, n[1] + dy))
        nodes = next_nodes
    return None


if __name__ == "__main__":
    """
    for i in range(72):
        for j in range(5):
            print("habitat%d, inter %d" %(i, j), end="\r")
            sp, ep = sample_start_end_position(i)
            print(sp, ep)
            if sp:
                crop_image_from_start_end(sp[0], sp[1], ep[0], ep[1], i, 0, time, recovery, total)
            # import pdb; pdb.set_trace()
    """
    import argparse

    parser = argparse.ArgumentParser(description = 'Start condor training')
    parser.add_argument('--habitat_index', dest='habitat_index', default=0)
    args = parser.parse_args()

    habitat_index = int(args.habitat_index)
    sp, ep = sample_start_goal_position(habitat_index)
    print(sp, ep)
    with open("configs/dwa_habitat.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    env_config = config["env_config"]
    env_config["kwargs"]["init_position"] = [*sp, np.arctan2( ep[1] - sp[1], ep[0] - sp[0])]
    env_config["kwargs"]["goal_position"] = [ep[0] - sp[0], ep[1] - sp[1], 0]
    env_config["kwargs"]["world_name"] = "habitat%d.sdf" %(habitat_index)
    print(env_config["kwargs"]["init_position"], env_config["kwargs"]["goal_position"])

    env = gym.make(env_config["env_id"], **env_config["kwargs"])

    env.reset()
    done = False
    start_pos = env.gazebo_sim.get_model_state().pose.position
    traj_pos = [(start_pos.x, start_pos.y)]
    start_time = rospy.get_time()

    spots = []

    while not done:
        _, _, done, info = env.step(env_config["kwargs"]["param_init"])
        robot_pos = env.gazebo_sim.get_model_state().pose.position
        traj_pos.append((robot_pos.x, robot_pos.y))
        if elucidate_distance(robot_pos, start_pos) >= SIZE:  # or done:
            end_pos = robot_pos
            end_time = rospy.get_time()
            time = end_time - start_time
            recovery, total = env.move_base.get_bad_vel_num()

            crop_image_from_start_end(start_pos.x, start_pos.y, end_pos.x, end_pos.y, habitat_index, time, recovery, total, traj_pos)

            start_pos = end_pos
            start_time = end_time
            env.move_base.reset_vel_count()
    env.close()