import numpy as np
from scipy.spatial.transform import Rotation as R

from autolab_core import (
    CameraChessboardRegistration,
    RigidTransform,
    YamlConfig,
)

from perception import RgbdSensorFactory
import pyrealsense2 as rs

def discover_cams():
    """Returns a list of the ids of all cameras connected via USB."""
    ctx = rs.context()
    ctx_devs = list(ctx.query_devices())
    ids = []
    for i in range(ctx.devices.size()):
        ids.append(ctx_devs[i].get_info(rs.camera_info.serial_number))
    return ids

config_filename = "cfg/tools/camera_registration.yaml"
config = YamlConfig(config_filename)

ids = discover_cams()
assert ids, "[!] No camera detected."
cfg = {}
cfg["cam_id"] = ids[0]
cfg["filter_depth"] = True
cfg["frame"] = "realsense"

sensor = RgbdSensorFactory.sensor("realsense", cfg) 
sensor.start()

registration_config = config["sensors"]["realsense"]["registration_config"].copy()
registration_config.update(config["chessboard_registration"])
sensor.ir_frame = sensor.frame
sensor.ir_intrinsics = sensor.color_intrinsics
reg_result = CameraChessboardRegistration.register(
    sensor, registration_config
)
T_camera_cb = reg_result.T_camera_cb
T_gripper_cb = RigidTransform(rotation=np.diag([-1,1,-1]), from_frame="gripper", to_frame="cb")
T_gripper_world = RigidTransform.load("cfg/T_gripper_world.tf")
T_realsense_world = T_gripper_world * T_gripper_cb.inverse() * T_camera_cb

print(T_realsense_world)
print(T_realsense_world.euler_angles)
T_realsense_world.save("cfg/T_realsense_world.tf")
