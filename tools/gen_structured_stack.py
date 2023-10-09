from autolab_core import RigidTransform
from collections import defaultdict
from datetime import date
from franka_base_polymetis import FrankaBase, save_data
from typing import List
import argparse
import datetime
import fcl
import numpy as np
import os
import time
import trimesh

def gen_random_pos(rt, xy_rand, z_rand, z_offset=0):
    """
    Generate a random position within a box around the center of rt
    """
    new_rt = rt.copy()
    pos = new_rt.translation
    pos[0] += np.random.uniform(-xy_rand, xy_rand)
    pos[1] += np.random.uniform(-xy_rand, xy_rand)
    pos[2] += np.random.uniform(-z_rand, z_rand) + z_offset
    return new_rt

class FrankaCubeStack(FrankaBase):
    def __init__(
        self, 
        robot_ip : str = "10.0.0.1", 
        wrist_cam_id : tuple = (4, 0, 8), # left, hand, right camera ids
        above_bin : str = "cfg/pre_grasp.tf",
        data_dir : str = "data",
        num_grasps = 200,
        vis=False,
        gripper_blocking=False,
        cube_size=0.054,
        z_height=0.015,
    ) -> None:
        super().__init__(
            robot_ip, 
            wrist_cam_id, 
            above_bin, 
            data_dir, 
            gripper_blocking_data_sync=gripper_blocking
        )
        self.cube_size = cube_size
        self.z_height = z_height
        self.vis = vis
        # object_poses is a dictionary of object name and its pose
        self.object_poses = defaultdict(list)

        # define boundary conditions 
        self.top_left = RigidTransform.load("cfg/top_left.tf")
        self.bottom_right = RigidTransform.load("cfg/bottom_right.tf")
        
        # determine a feeding position, and move to that position
        self.canonical_rot_mat = np.diag([1, -1, -1])

        # feed all objects 
        self.feed_objects(z_height=self.z_height)

        # generate all grasps based on boundary conditions, num_grasps and num_rotations 
        self.num_grasps = num_grasps
        self.gen_grasps(
            z_height=self.z_height,
            cube_dim=np.ones(3) * self.cube_size,
        )
        self.xy_rand = 0.02

    def gen_random_pose(
        self, 
        bad_poses : List[RigidTransform] = [], 
        z_rotation_range : tuple = (-np.pi/4, np.pi/4), 
        z_height : float = 0.015,
        object_dim : tuple = (0.054, 0.054, 0.054),
    ):
        """
        
        Args:
            bad_poses (list of RigidTransform): a list of poses that cannot overlap with 

        Returns:
            RigidTransform: a pose that satisfy the condition
        """
        
        # generate collision objects for the bad_poses
        box_dim = np.array([0.07, 0.23, 0.05])
        object_dim = np.array(object_dim)
        other_objects = []
        for pose in bad_poses:
            t = fcl.Transform(pose.rotation, pose.translation)
            b = fcl.Box(*object_dim)
            obj = fcl.CollisionObject(b, t)
            other_objects.append(obj)
        
        collision_manager = fcl.DynamicAABBTreeCollisionManager()
        collision_manager.registerObjects(other_objects)
        collision_manager.setup()
        
        current_object = fcl.CollisionObject(fcl.Box(*box_dim), fcl.Transform())
        collided = True 
        while collided:
            candidate_pos = np.random.uniform(low=self.top_left.translation[:2], high=self.bottom_right.translation[:2])
            candidate_pos = np.append(candidate_pos, z_height)
            candidate_pose = RigidTransform(
                rotation=self.canonical_rot_mat,
                translation=candidate_pos,
                from_frame="gripper",
                to_frame="world"
            ) * RigidTransform(
                rotation=RigidTransform.z_axis_rotation(np.random.uniform(low=z_rotation_range[0], high=z_rotation_range[1])),
                from_frame="gripper",
                to_frame="gripper",
            )
            current_object.setRotation(candidate_pose.rotation)
            current_object.setTranslation(candidate_pose.translation)
            
            # check collision 
            req = fcl.CollisionRequest(num_max_contacts=100, enable_contact=True)
            rdata = fcl.CollisionData(request = req)
            collision_manager.collide(current_object, rdata, fcl.defaultCollisionCallback)
            collided = rdata.result.is_collision
        
        if self.vis: 
            # all other objects are in green, the object of interest is in red
            green = np.array([0, 255, 0, 255])
            red = np.array([255, 0, 0, 255])
            meshes = []
            for pose in bad_poses:
                m = trimesh.creation.box(extents=object_dim, transform=pose.matrix)
                m.visual.vertex_colors = green
                meshes.append(m)
            m = trimesh.creation.box(extents=box_dim, transform=candidate_pose.matrix)
            m.visual.vertex_colors = red
            meshes.append(m)
            trimesh.Scene(meshes).show()
                
        return candidate_pose
    
    def feed_objects(self, z_height=0.015, radius=0.20):
        """
        feed the objects to the robot.
        """
        
        feed_pose = self.gen_random_pose(
            bad_poses=[p[-1] for p in self.object_poses.values() if p], 
            z_height=z_height, 
            object_dim=np.ones(3) * self.cube_size,
        )
        
        self.open_gripper()
        self.goto_pose(feed_pose, z_offset=0.10)
        self.goto_pose(feed_pose, z_offset=0.03)
        
        # feed the object 
        object_name = input("Enter object name: ")
        self.goto_pose(feed_pose)
        self.close_gripper()
        self.open_gripper()
        self.goto_pose(feed_pose, z_offset=0.10)
        
        # record object and its current pose
        self.object_poses[object_name].append(feed_pose)
        
        # prompt the user whether there are more objects to feed
        more_object = input("More objects to feed? (y/n)")
        if more_object == "y":
            self.feed_objects(z_height, radius)
        
    def gen_grasps(self, z_height = 0.015, cube_dim = (0.054, 0.054, 0.054)):
        self.grasps = []
        # we sequentially generate grasps that are not colliding with each other
        self.object_names = sorted(self.object_poses.keys())

        for grasp_idx in range(self.num_grasps):
            # generate a random pose
            cur_object_name = self.object_names[grasp_idx % len(self.object_names)]
            
            target_object_name = np.random.choice([n for n in self.object_names if n != cur_object_name])
            stack_object_pose = self.object_poses[target_object_name][-1].copy()
            stack_object_pose.translation[2] += cube_dim[2]

            # get the latest pose for each object other than the selected one
            all_other_poses = [p[-1] for n, p in self.object_poses.items() if n != cur_object_name]
            candidate_place_pose = self.gen_random_pose(
                all_other_poses, 
                z_rotation_range=(-np.pi/4, np.pi/4), 
                z_height=z_height, 
                object_dim=cube_dim
            )
            # Pick action
            self.grasps.append(
                {
                    "action_type": "stack",
                    "object": cur_object_name,
                    "target": target_object_name,
                    "pick": self.object_poses[cur_object_name][-1], 
                    "place": stack_object_pose,
                }
            )
            self.object_poses[cur_object_name].append(stack_object_pose)
            
            # Place action
            self.grasps.append(
                {
                    "action_type": "destack",
                    "object": cur_object_name,
                    "target": "table",
                    "pick": self.object_poses[cur_object_name][-1], 
                    "place": candidate_place_pose,
                }
            )
            self.object_poses[cur_object_name].append(candidate_place_pose)
        return self.grasps
    
    def run_demo(self):
        for idx, poses in enumerate(self.grasps):
            print("---------- Trajectory: {:,d} ----------".format(idx))
            self.step(poses)
            time.sleep(1)

    def step(self, action):
        action_type, object_name, target_name, grasp_pose, next_grasp_pose = action["action_type"], action["object"], action["target"], action["pick"], action["place"]
        
        # set gripper status 
        self.gripper_closed = False

        # create directory for trajectory 
        data_dir = str(self.data_dir).split("/")
        data_dir[-1] = str(action_type) + "-" + data_dir[-1]
        data_dir = "/".join(data_dir)
        traj_fp = os.path.join(data_dir, datetime.datetime.now().isoformat())
        os.makedirs(traj_fp, exist_ok=True)

        # Move to pregrasp positions 
        self.goto_pre_grasp(noise=5e-2)

        # Start recording, Execute grasp, stop recording
        self.init_data_sync()

        self.init_impedance_control()

        # print("Goto above grasp")
        self.goto_poses(
            [grasp_pose, grasp_pose], 
            z_offset=[np.random.uniform(0.06, 0.15), 0.0],
        )

        # Grasp the object, lift and drop it 
        self.close_gripper()
        
        # add randomization to pre-grasp
        place_xy_rand = next_grasp_pose.copy()
        place_xy_rand = gen_random_pos(place_xy_rand, self.xy_rand, 0.0)

        self.goto_poses(
            [grasp_pose, place_xy_rand, next_grasp_pose], 
            z_offset=[np.random.uniform(0.06, 0.15), np.random.uniform(0.06, 0.15), 0.0]
        )    
        
        # get gripper status to know if the grasp succeeded
        current_width = self.get_gripper_width()
        if current_width < 0.0003:
            self.grasp_suc = False 
        else:
            self.grasp_suc = True
        self.open_gripper()
        
        self.goto_pose(next_grasp_pose, z_offset=0.3)
        self.end_impedance_control()

        # Stop the recording thread
        recorded_data = self.end_data_sync()

        # metadata
        info = {
            "success": self.grasp_suc,
            "task": action_type,
            "object": object_name,
            "target": target_name, 
            "distractor": [n for n in self.object_names if n != object_name]
        }

        # Start Saving Thread
        save_data(traj_fp, recorded_data, self.grasp_suc, info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--dir_name", default=date.today().strftime("%m-%d-%Y"), type=str)
    parser.add_argument("--num_demos", default=3, type=int)
    parser.add_argument("--cube_size", default=0.054, type=float)
    parser.add_argument("--z_height", default=0.015, type=float)
    parser.add_argument("--gripper_blocking", default=False, action="store_true", help="don't record data while closing gripper")
    parser.add_argument("--wrist_cam_id", default=(4, 0, 8), type="+", help="left, hand, right camera ids")
    parser.add_argument("--vis", default=False, type=bool)
    args = parser.parse_args()
    data_collection = FrankaCubeStack(
        data_dir=os.path.join(args.data_dir, args.dir_name), 
        num_grasps=args.num_demos, 
        vis=args.vis, 
        gripper_blocking=args.gripper_blocking,
        cube_size=args.cube_size,
        z_height=args.z_height,
        wrist_cam_id=args.wrist_cam_id,
    )
    data_collection.run_demo()
