#!/usr/bin/env python

import cv2
import argparse
import imgviz
import joblib
import os

def vis_traj(path, vertical):
    """Visualizes a trajectory."""

    for fname in sorted(os.listdir(path)):
        if not fname.endswith(".pkl"):
            continue
        with open(os.path.join(path, fname), "rb") as f:
            data = joblib.load(f)
            # print(list(data.keys()))
            print(list(data["time_stamps"]))
            print(list(data["joint_pos"]))
        shape = (3, 1) if vertical else (1, 3)
        im_tile = imgviz.tile(
            [data["rgb_left"], data["rgb_hand"], data["rgb_right"]],
            shape=shape,
            border=(255, 255, 255)
        )
        imgviz.io.cv_imshow(im_tile)
        key = imgviz.io.cv_waitkey(1000 // (30 * 4))
        if key == ord("q"):
            break
        if key == ord("s"):
            cv2.imwrite("{}.png".format(fname), cv2.cvtColor(im_tile, cv2.COLOR_BGR2RGB))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-dir", dest="traj_dir", required=True)
    parser.add_argument("--vertical", action="store_true", default=False)
    args = parser.parse_args()
    vis_traj(args.traj_dir, args.vertical)
