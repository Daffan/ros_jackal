BASE_WORLD_PATH = "scripts/base_world.world"

def moving_cylinder(name, t1, x1, y1, t2, x2, y2):
    return "\
    <actor name=\"%s\">\n\
      <link name=\"box_link\">\n\
        <visual name=\"visual\">\n\
          <geometry>\n\
            <cylinder>\n\
              <radius>0.075000</radius>\n\
              <length>0.8</length>\n\
            </cylinder>\n\
          </geometry>\n\
        </visual>\n\
      </link>\n\
      <script>\n\
        <loop>true</loop>\n\
        <auto_start>true</auto_start>\n\
        <trajectory id=\"0\" type=\"square\">\n\
           <waypoint>\n\
              <time>%.2f</time>\n\
              <pose>%.2f %.2f 0 0 0 0</pose>\n\
           </waypoint>\n\
           <waypoint>\n\
              <time>%.2f</time>\n\
              <pose>%.2f %.2f 0 0 0 0</pose>\n\
           </waypoint>\n\
        </trajectory>\n\
      </script>\n\
    </actor>\n\
" %(name, t1, x1, y1, t2, x2, y2)

def sample_waypoints(direction, speed, idx):
    assert direction in DIRECTIONS, "direction %s not defined!" %(direction)
    x1, y1 = BOTTOM_LEFT
    x2, y2 = TOP_RIGHT
    if direction == "B2T":
        start_x = np.random.random() * (x2 - x1) + x1
        start_y = y1
        end_x = np.random.random() * (x2 - x1) + x1
        end_y = y2
    elif direction == "T2B":
        end_x = np.random.random() * (x2 - x1) + x1
        end_y = y1
        start_x = np.random.random() * (x2 - x1) + x1
        start_y = y2
    elif direction == "R2L":
        start_x = x1
        start_y = np.random.random() * (y2 - y1) + y1
        end_x = x2
        end_y = np.random.random() * (y2 - y1) + y1
    elif direction == "L2R":
        end_x = x1
        end_y = np.random.random() * (y2 - y1) + y1
        start_x = x2
        start_y = np.random.random() * (y2 - y1) + y1

    distance = ((start_x - end_x) ** 2 + (start_y - end_y) ** 2) ** 0.5
    time = distance / speed
    return "%s_%.1f_%d" %(direction, speed, idx), 0, start_x, start_y, time, end_x, end_y

BOTTOM_LEFT = (-4.5, 5)
TOP_RIGHT = (0, 9.5)

DIRECTIONS = ["R2L", "L2R", "T2B", "B2T"]
SPEEDS = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]
 
MIN_NUM_MOVING_OBJECT = 3
MAX_NUM_MOVING_OBJECT = 9

if __name__ == "__main__":
    import argparse
    import numpy as np
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="jackal_helper/worlds/BARN")
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--min_speed', type=float, default=0.3)
    parser.add_argument('--max_speed', type=float, default=1.9)
    parser.add_argument('--min_object', type=int, default=3)
    parser.add_argument('--max_object', type=int, default=9)
    parser.add_argument('--start_idx', type=int, default=300)
    parser.add_argument('--n_worlds', type=int, default=100)
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    
    with open(BASE_WORLD_PATH, "r") as f:
        ss = f.read()
        part1 = ss.split("TOKEN")[0]
        part2 = ss.split("TOKEN")[1]
        
    for i in range(args.n_worlds):
        mid = ""
        for j in range(np.random.randint(args.min_object, args.max_object)):
            waypoints = sample_waypoints(np.random.choice(DIRECTIONS), np.random.uniform(args.min_speed, args.max_speed), j)
            mid += moving_cylinder(*waypoints)
            
        with open(os.path.join(args.save_dir, "world_%d.world" %(i + args.start_idx)), "w") as f:
            f.write(part1 + mid + part2)