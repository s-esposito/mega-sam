import sys

sys.path.append("base/droid_slam")

from tqdm import tqdm
import numpy as np
import torch
import cv2
import os
import glob
import argparse
from lietorch import SE3

import torch.nn.functional as F
from droid import Droid

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", required=True)
    parser.add_argument("--scene_name", help="scene_name", required=True)
    parser.add_argument(
        "--mono_depth_path", default="Depth-Anything/video_visualization"
    )
    parser.add_argument("--metric_depth_path", default="UniDepth/outputs")
    parser.add_argument("--cvd_path", required=True)
    parser.add_argument("--vis_outputs_path", default="outputs_vis")
    args = parser.parse_args()

    print(args)
    
    # check if cvd_path exists (${seq}_sgd_cvd_hr.npz)
    cvd_path = os.path.join(args.cvd_path, args.scene_name) + "_sgd_cvd_hr.npz"
    if not os.path.exists(cvd_path):
        raise ValueError(f"{cvd_path} does not exist!")
    
    # check if mono_depth_path exists
    mono_depth_path = os.path.join(args.mono_depth_path, args.scene_name)
    if not os.path.exists(mono_depth_path):
        raise ValueError(f"{mono_depth_path} does not exist!")
    
    # check if metric_depth_path exists
    metric_depth_path = os.path.join(args.metric_depth_path, args.scene_name)
    if not os.path.exists(metric_depth_path):
        raise ValueError(f"{metric_depth_path} does not exist!")
    
    # if vis_outputs_path does not exist, create
    vis_outputs_path = os.path.join(args.vis_outputs_path, args.scene_name)
    if not os.path.exists(vis_outputs_path):
        os.makedirs(vis_outputs_path)

    # load cvd
    cvd = np.load(cvd_path)
    
    images = cvd["images"]
    depths = cvd["depths"]
    intrinsic = cvd["intrinsic"]
    cam_c2w = cvd["cam_c2w"]
    
    #
    img = images[0]
    
    # get max and min depth
    min_depth = np.min(depths)
    max_depth = np.max(depths)
    
    frame_idx = 0
    for image, depth in zip(images, depths):
    
        # plot side by side img and depth
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        im = axs[0].imshow(image)
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cax.axis("off")  # hide colorbar
        axs[0].set_title("Image")
        im = axs[1].imshow(depth, cmap="plasma", vmin=min_depth, vmax=max_depth)
        # add colorbar
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        axs[1].set_title("Depth")
        frame_idx_str = str(frame_idx).zfill(4)
        # save fig
        plt.savefig(
            os.path.join(vis_outputs_path, f"{frame_idx_str}.png"),
            transparent=False,
            bbox_inches="tight",
            pad_inches=1,
            dpi=100,
        )
        plt.close("all")
        
        frame_idx += 1