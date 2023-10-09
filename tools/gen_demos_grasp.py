from autolab_core import RigidTransform
from datetime import date
from franka_base_polymetis import FrankaBase, save_data
from grasp_planner import DexNetGraspPlanner
from perception import RgbdSensorFactory # TODO: deprecate this at some point
import argparse
import datetime
import numpy as np
import os
import time

class FrankaDexNet(FrankaBase):
    def __init__(
        self, 
        robot_ip : str = "10.0.0.1", 
        wrist_cam_id : tuple = (4, 0, 8), # left, hand, right camera ids
        depth_cam_id : str = "f1371550", 
        depth_cam_pose : str = "cfg/T_realsense_world.tf", 
        above_bin : str = "cfg/pre_grasp.tf",
        data_dir : str = "data",
        num_grasps = 200,
        waypoint_pos_noise_bound = np.ones(3) * 0.05, 
        waypoint_rot_noise_bound = np.ones(3) * np.pi/90,
        vis : bool = True,
        gripper_blocking=False,
    ) -> None:
        super().__init__(
            robot_ip, 
            wrist_cam_id, 
            above_bin, 
            data_dir, 
            waypoint_pos_noise_bound,
            waypoint_rot_noise_bound,
            gripper_blocking_data_sync=gripper_blocking,
        )
        
        self.vis = vis
        
        # move robot to home position
        self.open_gripper()
        self.goto_home()

        # initialize all the cameras 
        self.depth_camera_pose = RigidTransform.load(depth_cam_pose)
        self.init_realsense(depth_cam_id) # creates self.depth_sensor

        # initialize dexnet
        self.planer = DexNetGraspPlanner(visualize=vis)
        self.num_grasps = num_grasps
    
    def init_realsense(self, camera_id : str):
        cfg = {"cam_id": camera_id, "filter_depth": True, "frame": "realsense"}
        self.depth_sensor = RgbdSensorFactory.sensor("realsense", cfg) 
        self.depth_sensor.start()

    def run_demo(self):
        for idx in range(self.num_grasps):
            print("---------- Trajectory: {:,d} ----------".format(idx))
            self.step()
            time.sleep(1)

    def step(self, action = None): # the action will be determined when executing the actions
        # set gripper status 
        self.gripper_closed = False

        # create directory for trajectory 
        traj_fp = os.path.join(self.data_dir, datetime.datetime.now().isoformat())
        os.makedirs(traj_fp, exist_ok=True)

        # move to home position
        self.goto_home()

        # generate grasp based on dexnet 
        _, depth_im = self.depth_sensor.frames()

        depth_im = depth_im.data
        if len(depth_im.shape) == 2:
            depth_im = np.tile(depth_im[..., None], (1, 1, 4))
        grasp_pose = self.planer.plan_from_dimg(depth_im, self.depth_camera_pose)

        # Move to pregrasp positions 
        self.goto_pre_grasp(noise=5e-2)

        # Start recording, Execute grasp, stop recording
        self.init_data_sync()
        self.init_impedance_control()

        self.goto_poses(
            [grasp_pose, grasp_pose], 
            z_offset=[np.random.uniform(0.06, 0.15), 0.0],
        )

        # Grasp the object, lift and drop it 
        self.close_gripper()
        
        # Lifting the object to pregrasp pose 
        self.goto_pose(grasp_pose, z_offset=0.3)

        # Stop the recording thread
        self.end_impedance_control()
        recorded_data = self.end_data_sync()
        
        # get gripper status to know if the grasp succeeded
        current_width = self.get_gripper_width()
        if current_width < 0.0003:
            self.grasp_suc = False 
        else:
            self.grasp_suc = True
        
        # metadata
        info = {
            "success": self.grasp_suc,
            "task": "pick dexnet",
        }

        # Start Saving Thread
        save_data(traj_fp, recorded_data, self.grasp_suc, info)

        # randomly pick a drop location
        self.goto_pre_grasp(noise=5e-2)
        self.open_gripper()

        # Move to home position
        self.goto_home()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--dir_name", default=date.today().strftime("%m-%d-%Y"), type=str)
    parser.add_argument("--num_demos", default=40, type=int)
    parser.add_argument("--vis", default=False, type=bool)
    args = parser.parse_args()
    data_collection = FrankaDexNet(data_dir=os.path.join(args.data_dir, args.dir_name), num_grasps=args.num_demos, vis=args.vis)
    data_collection.run_demo()
