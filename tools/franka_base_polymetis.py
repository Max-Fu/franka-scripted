from autolab_core import RigidTransform
from autolab_core import transformations
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple
import joblib
import json
import numpy as np
import os
import signal
import threading
import time
import torch 

from polymetis import RobotInterface
from polymetis import GripperInterface
from utils import WebcamSensor, RobotDS, GripperExternDS, DataSync

from scipy.spatial.transform import Rotation, Slerp


class timeout:
    """
    borrowed from:
    https://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish
    """
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def flip_color(im):
    return im[:, :, ::-1]


def save_mp_data(idx, data_dir, rgb_left, rgb_hand, rgb_right, joint_pos, gripper_closed):
    out_f = os.path.join(data_dir, "{:04d}.pkl".format(idx))
    save_data = {
        "time_stamps": [rgb_left[1], rgb_hand[1], rgb_right[1], joint_pos[1], gripper_closed[1]],
        "joint_pos": joint_pos[0][:7],
        "joint_vel": joint_pos[0][7:14],
        "gripper_closed": gripper_closed[0],
        "rgb_left": flip_color(rgb_left[0]),
        "rgb_hand": flip_color(rgb_hand[0]),
        "rgb_right": flip_color(rgb_right[0])
    }
    joblib.dump(save_data, out_f, compress=3)

def save_data(data_dir : str, data : list, grasp_suc : bool, other_info : str = None, num_processes : int = 8):
    # Save data
    with Pool(num_processes) as p:
        p.starmap(save_mp_data, [(idx, data_dir, *d) for idx, d in enumerate(data)])
    if grasp_suc:
        open(os.path.join(data_dir, "success.txt"), "w").close()
    else:
        open(os.path.join(data_dir, "fail.txt"), "w").close()

    if other_info is not None:
        with open(os.path.join(data_dir, "metadata.json"), "w") as f:
            json.dump(other_info, f)

class FrankaBase:
    def __init__(
        self, 
        robot_ip : str = "10.0.0.1", 
        wrist_cam_id : tuple = (4, 0, 8), # left, hand, right camera ids
        above_bin : str = "cfg/pre_grasp.tf",
        data_dir : str = "data",
        waypoint_pos_noise_bound : np.ndarray = np.zeros(3), 
        waypoint_rot_noise_bound : np.ndarray = np.zeros(3),
        gripper_offset = 0.10, 
        vel=0.0015,
        data_sync_freq=30,
        gripper_blocking_data_sync=False,
        goto_error_thres=0.02,
    ) -> None:
        self.robot_ip = robot_ip
        self.init_robot()
        self.home_joints = np.array([-0.0402, -1.0500,  0.0175, -2.1226,  0.0185,  1.0450,  0.7980])
        self.pre_grasp_joints = np.array([-0.029610830371055684, 0.11720150547696832, -0.014184998175946244, -1.9407867052761583, -0.009267809184061155, 2.068323301143607, 0.8233103690213626])
        self.above_bin = RigidTransform.load(above_bin)

        # initialize all the cameras 
        self.robot_ds = RobotDS(self.robot, freq=1/60, max_buffer_len=5)
        self.gripper_ds = GripperExternDS(freq=1/60, max_buffer_len=5)
        self.datastreams = [WebcamSensor(cam_id, max_buffer_len=5) for cam_id in wrist_cam_id] + [self.robot_ds, self.gripper_ds] 

        # create data directory
        self.data_dir = data_dir

        # placeholder for data threads
        self.data_sync_thread = None
        self.data_sync = None

        # initialize gripper status 
        self.gripper_closed = False
        
        # initialize grasp success status 
        self.grasp_suc = False
        
        # initialize cartesian impedance controller 
        self.impedance_motion = None

        self.waypoint_pos_noise_bound = waypoint_pos_noise_bound
        self.waypoint_rot_noise_bound = waypoint_rot_noise_bound
        self.gripper_offset = gripper_offset
        
        self.gripper_default_rot = RigidTransform(
            rotation = RigidTransform.z_axis_rotation(np.pi/4),
            from_frame="gripper", 
            to_frame="gripper",
        )
        self.control_freq = 60.
        self.vel = vel 
        self.data_sync_freq = data_sync_freq
        self.gripper_blocking_data_sync = gripper_blocking_data_sync
        self.goto_error_thres = goto_error_thres

    def init_data_sync(self):
        self.data_read_threads = [threading.Thread(target=dt.start_read, daemon=True) for dt in self.datastreams]
        self.data_sync = DataSync(self.datastreams, frequency=1/self.data_sync_freq)
        self.data_sync_thread = threading.Thread(target=self.data_sync.start_ros, daemon=True)
        [i.start() for i in self.data_read_threads]
        while not all([len(dt.buffer)==dt.max_buffer_len for dt in self.datastreams]):
            time.sleep(0.1)
        self.data_sync_thread.start()
            
    def end_data_sync(self):
        recorded_data = self.data_sync.stop()
        self.data_sync = None
        self.data_sync_thread.join()
        time.sleep(0.1)
        [i.end_read() for i in self.datastreams]
        [i.join() for i in self.data_read_threads]
        return recorded_data

    def set_gains(self):
        self.robot.Kq_default = torch.Tensor([200, 200, 200, 200, 200, 100, 100])
        self.robot.Kqd_default = torch.Tensor([5, 3, 5, 5, 5, 3, 3]) * 2
        self.robot.Kx_default = torch.Tensor([750, 750, 750, 30, 30, 30])

    def init_robot(self):
        self.robot = RobotInterface(
            ip_address=self.robot_ip,
            time_to_go_default=1.5
        )
        self.set_gains()
        
        self.gripper = GripperInterface(
            ip_address=self.robot_ip,
        )
        self.max_gripper_width = self.gripper.metadata.max_width
        self.controller_restart = 0
        with timeout(seconds=8):
            self.gripper.goto(width=0.0, speed=0.2, force=0)
            while self.get_gripper_width() > 0.02:
                time.sleep(1)
            self.gripper.goto(width=0.08, speed=0.2, force=0)
            while self.get_gripper_width() < 0.075:
                time.sleep(1)

    def goto_joint_positions(self, joint_positions):
        state_log = []
        error = float("inf")
        while not state_log or error > self.goto_error_thres:
            state_log = self.robot.move_to_joint_positions(torch.tensor(joint_positions))
            error = np.linalg.norm(self.get_joint_positions() - joint_positions) / 7
        return state_log
    
    def get_joint_positions(self):
        return self.robot.get_joint_positions().numpy()
    
    def goto_home(self): 
        self.goto_joint_positions(self.home_joints)

    def goto_pre_grasp(self, noise=None):
        if noise is None:
            self.goto_joint_positions(self.pre_grasp_joints)
        elif isinstance(noise, float):
            self.goto_joint_positions(self.pre_grasp_joints + np.random.uniform(-noise, noise, size=self.pre_grasp_joints.shape))
        else:
            raise NotImplementedError

    def open_gripper(self):
        self.gripper_ds.open_gripper()
        if self.gripper_blocking_data_sync and self.data_sync:
            time.sleep(self.data_sync.f)
            self.data_sync.pause()

        self.gripper.goto(width=0.08, speed=0.2, force=0)
        while self.get_gripper_width() < 0.075:
            time.sleep(1)
        if self.gripper_blocking_data_sync and self.data_sync:
            self.data_sync.resume()

    def close_gripper(self, force=40, speed=0.1):
        self.gripper_ds.close_gripper()
        if self.gripper_blocking_data_sync and self.data_sync:
            time.sleep(self.data_sync.f)
            self.data_sync.pause()

        self.gripper.grasp(speed=speed, force=force, epsilon_inner=0.1, epsilon_outer=0.1)
        while self.get_gripper_width() > 0.075:
            time.sleep(1)
        if self.gripper_blocking_data_sync and self.data_sync:
            self.data_sync.resume()

    def get_gripper_width(self):
        gripper_state = self.gripper.get_state()
        return gripper_state.width

    def get_gripper_tf(self):
        ee_pos, ee_quat = self.robot.get_ee_pose()
        return self.get_rigid_transform(ee_pos, ee_quat)

    def get_ee_pose(self, tensor=False):
        if tensor:
            return self.robot.get_ee_pose()
        else:
            ee_pos, ee_quat = self.robot.get_ee_pose()
            return ee_pos.numpy(), ee_quat.numpy()

    def pose_diff(self, pose_1 : RigidTransform, pose_2 : RigidTransform):
        v1 = np.append(pose_1.translation, pose_1.quaternion)
        v2 = np.append(pose_2.translation, pose_2.quaternion)
        pose_diff = np.linalg.norm(v1[:3] - v2[:3])
        quat_diff = 1 - np.dot(v1[3:], v2[3:])
        return pose_diff + quat_diff

    def inject_noise(self, pose, waypoint_pos_noise_bound=None, waypoint_rot_noise_bound=None):
        if waypoint_pos_noise_bound is None:
            waypoint_pos_noise_bound = self.waypoint_pos_noise_bound
        if waypoint_rot_noise_bound is None:
            waypoint_rot_noise_bound = self.waypoint_rot_noise_bound
        pos_noise = np.random.uniform(-waypoint_pos_noise_bound, waypoint_pos_noise_bound, size=3)
        rot_noise = np.random.uniform(-waypoint_rot_noise_bound, waypoint_rot_noise_bound, size=3)
        noisy_rotation = R.from_euler("xyz", rot_noise, degrees=False).as_matrix()
        noisy_pose = RigidTransform(
            rotation=noisy_rotation, 
            translation=pos_noise, 
            from_frame="gripper",
            to_frame="gripper",
        )
        return pose * noisy_pose

    # TODO make goto_pose_dynamic compatible with goto_pose (can call them consecutively)
    def goto_pose_dynamic(self, pose : RigidTransform, z_offset = 0.0, **kwargs):
        raise NotImplementedError
        

    def get_ee_pos_quat(self, pose : RigidTransform, use_tensor=True):
        """
        generate end effector position and quaternion from a RigidTransform object
        """
        pose = pose * self.gripper_default_rot
        ee_pos, ee_quat = pose.translation, transformations.quaternion_from_matrix(pose.matrix)
        if use_tensor:
            return torch.tensor(ee_pos).float(), torch.tensor(ee_quat).float()
        else:
            return ee_pos, ee_quat
        
    def get_rigid_transform(self, ee_pos, ee_quat, from_frame="gripper", to_frame="world"):
        """
        generate a RigidTransform object from end effector position and quaternion

        Args:
            ee_pos (torch.tensor): (3, )
            ee_quat (torch.tensor): (4, )
            from_frame (str, optional): _description_. Defaults to "gripper".
            to_frame (str, optional): _description_. Defaults to "world".

        Returns:
            RigidTransform: the rigid transform that describes the pose of the end effector
        """
        ee_pos, ee_quat = ee_pos.numpy(), ee_quat.numpy()
        rotation_mat = transformations.quaternion_matrix(ee_quat)[:3, :3]
        tf = RigidTransform(
            rotation = rotation_mat, 
            translation = ee_pos, 
            from_frame=from_frame, 
            to_frame=to_frame
        ) * self.gripper_default_rot.inverse()
        return tf

    def transform_interpolation(
        self,
        rt1 : RigidTransform, 
        rt2 : RigidTransform, 
    ):
        def smoothstep(x):
            return x * x * (3 - 2 * x)
            # return x * x * x * (x * (x * 6 - 15) + 10)
        pos_1, quat_1 = self.get_ee_pos_quat(rt1, use_tensor=False)
        pos_2, quat_2 = self.get_ee_pos_quat(rt2, use_tensor=False)
        
        trans_dist = np.linalg.norm(pos_1 - pos_2)
        rot_dist = 1 - np.dot(quat_1, quat_2) ** 2
        k = max(int(trans_dist / self.vel), int(rot_dist / self.vel))
        
        # Convert the input quaternions to Rotation objects
        rot1 = Rotation.from_quat(quat_1)
        rot2 = Rotation.from_quat(quat_2)

        # Create an array of k+1 equally spaced time points, including the start and end points
        times = np.linspace(0, 1, k + 1)
        smoothed_time = smoothstep(times)

        # Create a Slerp object for interpolation
        slerp = Slerp([0, 1], Rotation.from_rotvec(np.vstack((rot1.as_rotvec(), rot2.as_rotvec()))))

        # Interpolate the rotations for the specified time points
        interpolated_rots = slerp(smoothed_time)

        # Convert the interpolated rotations back to quaternions
        interpolated_quats = torch.tensor(np.array([rot.as_quat() for rot in interpolated_rots])).float()

        # Interpolate the positions
        interpolated_positions = np.array([pos_1 + t * (pos_2 - pos_1) for t in smoothed_time])
        positions = torch.tensor(interpolated_positions).float()
        
        return list(zip(positions, interpolated_quats))
    
    def multi_transform_interpolation(
        self,
        rts : List[RigidTransform],
    ):
        assert len(rts) >= 2, "need >= 2 waypoints to interpolate"
        if len(rts) == 2:
            return self.transform_interpolation(rts[0], rts[1])
        
        def smoothstep(x, segment_k, total_segments):
            x = np.clip((x + segment_k) / total_segments, 0, 1)
            y = 3 * x**2 - 2 * x**3
            return (y - y[0]) / (y[-1] - y[0])

        rts_pos_quat = [self.get_ee_pos_quat(rt, use_tensor=False) for rt in rts]
        interpolated_quats, interpolated_positions = [], []
        
        for segment_k in range(len(rts) - 1):
        
            pos_1, quat_1 = rts_pos_quat[segment_k]
            pos_2, quat_2 = rts_pos_quat[segment_k + 1]
                
            trans_dist = np.linalg.norm(pos_1 - pos_2)
            rot_dist = 1 - np.dot(quat_1, quat_2) ** 2
            total_segments = len(rts)-1
            vel = self.vel / total_segments * 2
            k = max(int(trans_dist / vel), int(rot_dist / vel))
            
            # Convert the input quaternions to Rotation objects
            rot1 = Rotation.from_quat(quat_1)
            rot2 = Rotation.from_quat(quat_2)

            # Create an array of k+1 equally spaced time points, including the start and end points
            times = np.linspace(0, 1, k + 1)
            smoothed_time = smoothstep(times, segment_k, total_segments)
            # Create a Slerp object for interpolation
            slerp = Slerp([0, 1], Rotation.from_rotvec(np.vstack((rot1.as_rotvec(), rot2.as_rotvec()))))

            # Interpolate the rotations for the specified time points
            interpolated_rots = slerp(smoothed_time)

            interpolated_quats.extend([rot.as_quat() for rot in interpolated_rots])
            interpolated_positions.extend([pos_1 + t * (pos_2 - pos_1) for t in smoothed_time])

        # Convert the interpolated rotations back to quaternions
        interpolated_quats = torch.tensor(np.array(interpolated_quats)).float()

        # Interpolate the positions
        positions = torch.tensor(np.array(interpolated_positions)).float()
        
        return list(zip(positions, interpolated_quats))

    def _move_to_ee_pose(self, ee_pos, ee_quat, op_space_interp=False):
        """
        Ensure ee_pos and ee_quat are reached
        """
        state_log = []
        error = float("inf")
        while not state_log or error > self.goto_error_thres:
            state_log = self.robot.move_to_ee_pose(
                position=ee_pos, orientation=ee_quat, op_space_interp=op_space_interp
            )
            error = self.pose_diff(self.get_gripper_tf(), self.get_rigid_transform(ee_pos, ee_quat))
        return state_log

    def _update_desired_ee_pose(self, position, orientation):
        """
        Ensure ee_pos and ee_quat are reached
        """
        try:
            state_log = self.robot.update_desired_ee_pose(position=position, orientation=orientation)
        except:
            print('impedance controller failed, restarting it')
            self.controller_restart += 1
            print(f'controller reset tracker: {self.controller_restart}\n')
            self.init_impedance_control(forced=True)
            state_log = self.robot.update_desired_ee_pose(position=position, orientation=orientation)
        return state_log

    def _update_desired_joint_positions(
        self,
        joint_positions : torch.Tensor, 
    ):
        try: 
            state_log = self.robot.update_desired_joint_positions(joint_positions)
        except:
            print('impedance controller failed, restarting it')
            self.controller_restart += 1
            print(f'controller reset tracker: {self.controller_restart}\n')
            self.init_impedance_control(forced=True)
            state_log = self.robot.update_desired_joint_positions(joint_positions)
        return state_log
    
    def _calculate_ik(
        self, 
        ee_pos : torch.Tensor, 
        ee_quat : torch.Tensor,
    ) -> Tuple[torch.Tensor, bool]: 
        return self.robot.solve_inverse_kinematics(ee_pos, ee_quat, self.robot.get_joint_positions())

    def goto_pose(self, pose : RigidTransform, z_offset = 0.0, **kwargs):
        """
        Blocking Action
        Move to a pose with rigid waypoints
        radius_thres, time_thres are not used
        note: z_offset is regarded as in world frame (z axis is pointing up). However, to account for different approach poses, 
        the z_offset is processed in gripper frame (z axis pointing out from gripper)
        """
        # need to make sure the velocity is 0 
        z_offset += self.gripper_offset
        approach_tf = RigidTransform(
            translation=np.array([0, 0, -z_offset]), 
            from_frame="gripper",
            to_frame="gripper",
        )
        updated_pose = pose * approach_tf
        if self.impedance_motion is None: 
            ee_pos, ee_quat = self.get_ee_pos_quat(updated_pose)
            state_log = self._move_to_ee_pose(
                ee_pos, ee_quat, op_space_interp=False
            )
        else:
            # calculate linear interpolation of k waypoints at particular control frequency
            pos_quats = self.transform_interpolation(self.get_gripper_tf(), updated_pose)
            start_time = time.time()
            for ee_pos, ee_quat in pos_quats:
                state_log = self._update_desired_ee_pose(ee_pos, ee_quat)
                time.sleep(max(1.0 / (self.control_freq * 2), 1.0 / self.control_freq - (time.time() - start_time)))
                start_time = time.time()

        return state_log
    
    def goto_poses(self, poses : list, z_offset = 0.0, **kwargs):
        """
        Blocking Action
        Move to a pose with rigid waypoints
        radius_thres, time_thres are not used
        note: z_offset is regarded as in world frame (z axis is pointing up). However, to account for different approach poses, 
        the z_offset is processed in gripper frame (z axis pointing out from gripper)
        """
        # need to make sure the velocity is 0 
        if type(z_offset) == float:
            z_offsets = [z_offset] * len(poses)
        else:
            assert len(z_offset) == len(poses)
            z_offsets = z_offset.copy()
        z_offsets = np.array(z_offsets)
        z_offsets += self.gripper_offset
        approach_tfs = [RigidTransform(
            translation=np.array([0, 0, -z_offset]), 
            from_frame="gripper",
            to_frame="gripper",
        ) for z_offset in z_offsets]
        updated_poses = [pose * approach_tf for pose, approach_tf in zip(poses, approach_tfs)]
        if self.impedance_motion is None: 
            poses_to_go = [self.get_ee_pos_quat(updated_pose) for updated_pose in updated_poses]
            for ee_pos, ee_quat in poses_to_go:
                state_log = self._move_to_ee_pose(
                    ee_pos, ee_quat, op_space_interp=False
                )
        else:
            # calculate linear interpolation of k waypoints at particular control frequency
            pos_quats = self.multi_transform_interpolation([self.get_gripper_tf()] + updated_poses)
            start_time = time.time()
            for ee_pos, ee_quat in pos_quats:
                state_log = self._update_desired_ee_pose(ee_pos, ee_quat)
                time.sleep(max(1.0 / (self.control_freq * 2), 1.0 / self.control_freq - (time.time() - start_time)))
                start_time = time.time()

        return state_log

    def init_impedance_control(self, forced=False):
        if forced or self.impedance_motion is None:
            self.robot.start_cartesian_impedance()
            self.impedance_motion = True 
            
    def end_impedance_control(self, forced=False):
        if forced or self.impedance_motion is not None:
            self.robot.terminate_current_policy()
            self.impedance_motion = None
    
    def run_demo(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
