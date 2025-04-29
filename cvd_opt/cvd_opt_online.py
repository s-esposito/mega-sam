import os
import torch
import numpy as np
from torch import nn
import torch.optim as optim

# Assume you have RAFT or any optical flow estimator ready
from raft import RAFT  # You would have to load a pretrained RAFT
from core.utils.utils import InputPadder

# Some utility modules you will need
from geometry_utils import NormalGenerator
import kornia

# ---- Parameters ----
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_ITERS = 50  # optimization iterations per new frame
FLOW_MODEL_PATH = 'raft-things.pth'  # pretrained RAFT checkpoint

# Loss Weights
W_FLOW = 0.2
W_RATIO = 1.0
W_GRAD = 2.0
W_NORMAL = 4.0

# Store previous frames
past_depths = []
past_poses = []
past_images = []

# Load RAFT model
raft_args = Namespace(small=False, mixed_precision=False, model=FLOW_MODEL_PATH)
flow_model = nn.DataParallel(RAFT(raft_args))
flow_model.load_state_dict(torch.load(FLOW_MODEL_PATH))
flow_model = flow_model.module.to(DEVICE).eval()


def compute_flow(image1, image2):
    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)
    with torch.no_grad():
        flow_low, flow_up = flow_model(image1, image2, iters=20, test_mode=True)
    return flow_up


def consistency_losses(depth_t, depth_prev, flow, pose_t, pose_prev, K, K_inv, compute_normals):
    # Implement simplified versions: flow consistency, depth ratio, gradient, normal
    # Placeholder examples:
    loss_flow = torch.mean((flow ** 2))
    loss_depth_ratio = torch.mean(torch.abs(depth_t - depth_prev))
    loss_grad = torch.mean(torch.abs(torch.gradient(depth_t)[0]))
    loss_normal = torch.mean(torch.abs(depth_t - depth_prev))
    return loss_flow, loss_depth_ratio, loss_grad, loss_normal


def process_new_frame(image_t, depth_t, pose_t, K, K_inv, compute_normals):
    global past_depths, past_poses, past_images

    selected_indices = range(max(0, len(past_depths) - 3), len(past_depths))  # last 3 frames

    depth_t = depth_t.clone().detach().to(DEVICE).requires_grad_(True)
    optimizer = optim.Adam([depth_t], lr=1e-3)

    for _ in range(NUM_ITERS):
        optimizer.zero_grad()
        total_loss = 0.0

        for idx in selected_indices:
            depth_prev = past_depths[idx]
            pose_prev = past_poses[idx]
            image_prev = past_images[idx]

            flow = compute_flow(image_t, image_prev)
            loss_flow, loss_ratio, loss_grad, loss_normal = consistency_losses(
                depth_t, depth_prev, flow, pose_t, pose_prev, K, K_inv, compute_normals
            )

            total_loss += (
                W_FLOW * loss_flow +
                W_RATIO * loss_ratio +
                W_GRAD * loss_grad +
                W_NORMAL * loss_normal
            )

        total_loss.backward()
        optimizer.step()

    # Save to memory
    past_depths.append(depth_t.detach())
    past_poses.append(pose_t)
    past_images.append(image_t)


def main():
    # Assume a dummy intrinsics matrix
    K = torch.eye(3).to(DEVICE)
    K[0, 0] = K[1, 1] = 500.0
    K[0, 2] = K[1, 2] = 256.0
    K_inv = torch.linalg.inv(K)

    # Normal computation utility
    compute_normals = [NormalGenerator(512, 512)]

    # Simulate incoming frames (you'd replace this with real data)
    for frame_idx in range(100):
        image_t = torch.randn(1, 3, 512, 512).to(DEVICE)  # Dummy random image
        depth_t = torch.abs(torch.randn(1, 512, 512)).to(DEVICE)  # Dummy random depth
        pose_t = torch.eye(4).to(DEVICE)  # Dummy identity pose (replace with real poses)

        process_new_frame(image_t, depth_t, pose_t, K, K_inv, compute_normals)

        print(f"Processed frame {frame_idx}")


if __name__ == "__main__":
    main()
