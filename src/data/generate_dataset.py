import os
import sys
import pickle
import random

import torch
import numpy as np

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

sys.path.append(ROOT_DIR)

from src.data.renderer import TensorRenderer

import yaml


# --------------------------------------------------
# load params
# --------------------------------------------------
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

IMAGE_SIZE = params["dataset"]["image_size"]

TRAIN_SPLIT = params["dataset"]["train_split"]
VAL_SPLIT = params["dataset"]["val_split"]
TEST_SPLIT = params["dataset"]["test_split"]

X_LIM = tuple(params["renderer"]["xlim"])
Y_LIM = tuple(params["renderer"]["ylim"])
ARM_RADIUS = params["renderer"]["arm_radius"]
TARGET_SIZE = params["renderer"]["target_size"]

SEED = params["seed"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# --------------------------------------------------
# load trajectories
# --------------------------------------------------
RAW_PATH = "data/raw/splined_trajectories_3.txt"

with open(RAW_PATH, "rb") as f:
    trajectories = pickle.load(f)

print(f"Loaded {len(trajectories)} trajectories")


# --------------------------------------------------
# initialize renderer
# --------------------------------------------------
renderer = TensorRenderer(
    image_size=IMAGE_SIZE,
    xlim=X_LIM,
    ylim=Y_LIM,
    arm_radius=ARM_RADIUS,
    target_size=TARGET_SIZE
)


# --------------------------------------------------
# generate tensors
# --------------------------------------------------
all_images = []
all_coords = []
all_targets = []

for traj in trajectories:

    traj = np.array(traj)

    left = traj[:, :2]
    right = traj[:, 2:]

    # determine target
    if np.max(np.linalg.norm(left - left[0], axis=1)) > \
       np.max(np.linalg.norm(right - right[0], axis=1)):

        target = left[-1]

    else:
        target = right[-1]

    traj_images = []

    for t in range(len(traj)):

        coords = traj[t]

        img = renderer.render(coords, target)

        traj_images.append(img)

    traj_images = torch.stack(traj_images)

    traj_coords = torch.tensor(
        traj,
        dtype=torch.float32
    )

    traj_target = torch.tensor(
        target,
        dtype=torch.float32
    )

    all_images.append(traj_images)
    all_coords.append(traj_coords)
    all_targets.append(traj_target)


# --------------------------------------------------
# stack dataset
# --------------------------------------------------
all_images = torch.stack(all_images)
all_coords = torch.stack(all_coords)
all_targets = torch.stack(all_targets)

print("Images:", all_images.shape)
print("Coords:", all_coords.shape)
print("Targets:", all_targets.shape)


# --------------------------------------------------
# split dataset
# --------------------------------------------------
N = len(all_images)

indices = list(range(N))
random.shuffle(indices)

train_end = int(TRAIN_SPLIT * N)
val_end = train_end + int(VAL_SPLIT * N)

train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]


def make_split(idxs):

    return {
        "images": all_images[idxs],
        "coords": all_coords[idxs],
        "targets": all_targets[idxs]
    }


train_data = make_split(train_idx)
val_data = make_split(val_idx)
test_data = make_split(test_idx)


# --------------------------------------------------
# save processed tensors
# --------------------------------------------------
os.makedirs("data/processed", exist_ok=True)

torch.save(
    train_data,
    "data/processed/train.pt"
)

torch.save(
    val_data,
    "data/processed/val.pt"
)

torch.save(
    test_data,
    "data/processed/test.pt"
)

print("Saved processed datasets")