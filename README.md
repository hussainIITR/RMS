# RMS
import numpy as np
from PIL import Image
import os
import json
import heapq
import csv

# ==================================================
# CONFIG
# ==================================================

GRID_SIZE = 20
EPS = 1e-6
WALL_INFLATION = 1.15   # mild, stable

ROOT = r"C:\Users\Hussain Salar\OneDrive\Desktop\IITR\RMS\SynapseDrive_Dataset"
IMAGE_DIR = os.path.join(ROOT, "test", "images")
VELOCITY_DIR = os.path.join(ROOT, "test", "velocities")
ASSET_ROOT = os.path.join(ROOT, "assets")

OUTPUT_CSV = "submission.csv"

LABELS = {
    "floor": 0,
    "wall": 1,
    "hazard": 2,
    "start": 3,
    "goal": 4
}

# ==================================================
# TERRAIN MAP
# ==================================================

FILE_MAP = {
    "forest": {
        "floor": ["t0_dirt.png"],
        "wall": ["t0_tree.png"],
        "hazard": ["t0_puddle.png"],
        "start": ["t0_startship.png"],
        "goal": ["t0_goal.png"],
    },
    "lab": {
        "floor": ["t2_floor.png"],
        "wall": ["t2_plasma.png", "t2_wall.png"],
        "hazard": ["t2_glue.png"],
        "start": ["t2_drone.png"],
        "goal": ["t2_goal.png"],
    },
    "desert": {
        "floor": ["t1_sand.png"],
        "wall": ["t1_rocks.png", "t1_cacti.png"],
        "hazard": ["t1_quicksand.png"],
        "start": ["t1_rover.png"],
        "goal": ["t1_goal.png"],
    }
}

BASE_COST = {
    "forest": {0: 1.5, 2: 2.8, 3: 1.5, 4: 1.5},
    "lab":    {0: 1.0, 2: 3.0, 3: 1.0, 4: 1.0},
    "desert": {0: 1.2, 2: 3.7, 3: 1.2, 4: 1.2},
}

MOVES = {
    "u": (-1, 0),
    "d": (1, 0),
    "l": (0, -1),
    "r": (0, 1),
}

# ==================================================
# UTILS
# ==================================================

def mse(a, b):
    return np.mean((a - b) ** 2)

def crop_to_grid(img):
    H, W = img.shape[:2]
    H2 = (H // GRID_SIZE) * GRID_SIZE
    W2 = (W // GRID_SIZE) * GRID_SIZE
    y0 = (H - H2) // 2
    x0 = (W - W2) // 2
    return img[y0:y0+H2, x0:x0+W2]

# ==================================================
# GRID POST-PROCESSING
# ==================================================

def postprocess_grid(grid):
    g = grid.copy()

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if g[i, j] in (3, 4):
                continue

            wall_n = 0
            hazard_n = 0

            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i+dx, j+dy
                if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE:
                    if g[ni, nj] == 1:
                        wall_n += 1
                    if g[ni, nj] == 2:
                        hazard_n += 1

            if wall_n >= 2:
                g[i, j] = 1
            elif g[i, j] == 2 and hazard_n == 0:
                g[i, j] = 0

    return g

# ==================================================
# TERRAIN DETECTION
# ==================================================

def detect_terrain(img, th, tw):
    tile = img[10*th:11*th, 10*tw:11*tw]
    best, score = None, 1e18

    for t in FILE_MAP:
        p = os.path.join(ASSET_ROOT, t, FILE_MAP[t]["floor"][0])
        ref = Image.open(p).convert("RGB").resize((tw, th))
        ref = np.asarray(ref, dtype=np.float32) / 255.0
        s = mse(tile, ref)
        if s < score:
            score, best = s, t

    return best

# ==================================================
# TEMPLATE CACHE
# ==================================================

CACHE = {}

def get_templates(terrain, th, tw):
    key = (terrain, th, tw)
    if key in CACHE:
        return CACHE[key]

    T = {}
    for k, lab in LABELS.items():
        T[lab] = []
        for f in FILE_MAP[terrain][k]:
            p = os.path.join(ASSET_ROOT, terrain, f)
            img = Image.open(p).convert("RGB").resize((tw, th))
            T[lab].append(np.asarray(img, dtype=np.float32) / 255.0)

    CACHE[key] = T
    return T

# ==================================================
# IMAGE → GRID
# ==================================================

def image_to_grid(img, templates, th, tw):
    g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            tile = img[i*th:(i+1)*th, j*tw:(j+1)*tw]
            scores = {k: min(mse(tile, t) for t in v)
                      for k, v in templates.items()}
            g[i, j] = min(scores, key=scores.get)

    return g

# ==================================================
# VELOCITY (FIXED)
# ==================================================

def load_velocity(path):
    with open(path) as f:
        d = json.load(f)

    if isinstance(d, list):
        arr = np.array(d, float)
    elif isinstance(d, dict):
        for k in ["velocity", "vel", "speed"]:
            if k in d:
                arr = np.array(d[k], float)
                break
        else:
            arr = np.array(d.get("data", []), float)
    else:
        raise RuntimeError("Invalid velocity format")

    if arr.size != GRID_SIZE * GRID_SIZE:
        raise RuntimeError(f"Velocity size {arr.size} != 400")

    return arr.reshape(GRID_SIZE, GRID_SIZE)

# ==================================================
# A* PLANNER
# ==================================================

def astar(grid, vel, terrain):
    start = goal = None
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if grid[i,j] == 3: start = (i,j)
            if grid[i,j] == 4: goal  = (i,j)

    if start is None or goal is None:
        return None

    pq = [(0, start)]
    g = {start: 0}
    parent = {}

    def h(a,b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    while pq:
        _, (x,y) = heapq.heappop(pq)
        if (x,y) == goal:
            break

        for m,(dx,dy) in MOVES.items():
            nx, ny = x+dx, y+dy
            if not (0<=nx<GRID_SIZE and 0<=ny<GRID_SIZE):
                continue
            if grid[nx,ny] == 1:
                continue

            base = BASE_COST[terrain].get(grid[nx,ny],1.0)
            cost = base / max(vel[nx,ny], EPS)

            for dx2, dy2 in [(-1,0),(1,0),(0,-1),(0,1)]:
                ax, ay = nx+dx2, ny+dy2
                if 0<=ax<GRID_SIZE and 0<=ay<GRID_SIZE:
                    if grid[ax,ay] == 1:
                        cost *= WALL_INFLATION
                        break

            ng = g[(x,y)] + cost
            if (nx,ny) not in g or ng < g[(nx,ny)]:
                g[(nx,ny)] = ng
                parent[(nx,ny)] = ((x,y), m)
                heapq.heappush(pq, (ng + h((nx,ny), goal), (nx,ny)))

    if goal not in parent:
        return None

    path = []
    cur = goal
    while cur != start:
        cur, m = parent[cur]
        path.append(m)

    return "".join(reversed(path))

# ==================================================
# MAIN
# ==================================================

def main():
    rows = []
    files = sorted(f for f in os.listdir(IMAGE_DIR) if f.endswith(".png"))
    print(f"Processing {len(files)} maps\n")

    for i,f in enumerate(files,1):
        mid = f[:-4]
        print(f"[{i}] {mid}")

        try:
            img = Image.open(os.path.join(IMAGE_DIR,f)).convert("RGB")
            img = np.asarray(img,np.float32)/255
            img = crop_to_grid(img)

            th, tw = img.shape[0]//GRID_SIZE, img.shape[1]//GRID_SIZE
            terrain = detect_terrain(img, th, tw)
            T = get_templates(terrain, th, tw)
            grid = image_to_grid(img, T, th, tw)
            grid = postprocess_grid(grid)

            vel = load_velocity(os.path.join(VELOCITY_DIR, mid+".json"))
            path = astar(grid, vel, terrain)

            if path is None:
                rows.append((mid,"u"))
                print("  u\n")
            else:
                rows.append((mid,path))
                print(f"  {len(path)} steps\n")

        except Exception as e:
            print("  ERROR", e)
            rows.append((mid,"u"))

    with open(OUTPUT_CSV,"w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id","path"])
        w.writerows(rows)

    print("DONE")

if __name__ == "__main__":
    main()
