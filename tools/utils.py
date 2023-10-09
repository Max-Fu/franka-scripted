#!/usr/bin/env python3

from PIL import Image
import cv2 
import itertools
import numpy as np
import os
import threading 
import time

class DummyDS:
    """
    A dummy data source that append one {id} to the buffer 
    """
    def __init__(self, id, max_buffer_len=10, freq=1/60):
        self.id = id 
        self.buffer, self.timestamp = [], []
        self.max_buffer_len = max_buffer_len
        self.lock = threading.Lock()
        self.active = False
        self.freq = freq
        self.index_last = [0]

    def start_read(self):
        self.active = True
        while self.active:
            time.sleep(self.freq)
            ct = time.time()
            self.buffer.append(f"{self.id}-{ct}")
            self.timestamp.append(ct)
            # print(f"Dummy DS: {self.id}-{ct}-{len(self.buffer)}")
            if self.lock.acquire(blocking=False):
                if len(self.buffer) > self.max_buffer_len:
                    # print(f"Dummy DS: {self.id}-{ct}-{len(self.buffer)}-trimming")
                    del self.buffer[:-self.max_buffer_len], self.timestamp[:-self.max_buffer_len]
                self.index_last[0] = len(self.buffer) - 1
                self.lock.release()
    
    def end_read(self):
        self.active = False 
        self.buffer = []
        self.timestamp = []

class WebcamSensor:
    """
    A webcam sensor that supports multithreading 
    """
    def __init__(self, cam_id, cam_res = [640, 480], set_fps = 60, max_buffer_len=10) -> None:
        """
        cam_id: read from v4l2-ctl --list-devices
        cam_res: (width, height)
        set_fps: set the fps of the camera
        max_buffer_len: the maximum number of frames to store in the buffer
        """
        #os.system("v4l2-ctl -d /dev/video{} -c exposure_auto=1 -c exposure_auto_priority=0 -c exposure_absolute=100 -c saturation=60 -c gain=140 -c focus_auto=1".format(cam_id))
        self.cap = cv2.VideoCapture(cam_id)
        width, height = cam_res
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, height)
        self.cap.set(cv2.CAP_PROP_FPS, set_fps)
        self.active = False
        self.buffer = []
        self.timestamp = []
        self.index_last = [0]
        self.max_buffer_len = max_buffer_len
        self.lock = threading.Lock()

    def start_read(self):
        self.flush()
        self.active = True
        while self.active:
            ret, frame = self.cap.read()
            self.buffer.append(frame)
            self.timestamp.append(time.time())
            if self.lock.acquire(blocking=False):
                if len(self.buffer) > self.max_buffer_len:
                    del self.buffer[:-self.max_buffer_len], self.timestamp[:-self.max_buffer_len]
                self.index_last[0] = len(self.buffer) - 1
                self.lock.release()

    def end_read(self):
        self.active = False 
        self.buffer = []
        self.timestamp = []

    def flush(self, num_ims=5):
        for i in range(num_ims):
            _, _ = self.cap.read()

class RobotDS:
    """
    A thread for reading robot joint states and velocity data at a fixed frequency
    """
    def __init__(self, robot, max_buffer_len=10, freq=1/60):
        self.robot = robot
        self.buffer, self.timestamp = [], []
        self.max_buffer_len = max_buffer_len
        self.lock = threading.Lock()
        self.active = False
        self.freq = freq
        self.index_last = [0]

    def start_read(self):
        self.active = True
        while self.active:
            if self.freq is not None:
                time.sleep(self.freq)
            ct = time.time()
            self.buffer.append(
                np.append(
                    self.robot.get_joint_positions(), 
                    self.robot.get_joint_velocities()
                )
            )
            self.timestamp.append(ct)
            if self.lock.acquire(blocking=False):
                if len(self.buffer) > self.max_buffer_len:
                    del self.buffer[:-self.max_buffer_len], self.timestamp[:-self.max_buffer_len]
                self.index_last[0] = len(self.buffer) - 1
                self.lock.release()
    
    def end_read(self):
        self.active = False 
        self.buffer = []
        self.timestamp = []

class GripperExternDS:
    """
    A thread for reading gripper width
    """
    def __init__(self, max_buffer_len=10, freq=1/60):
        self.buffer, self.timestamp = [], []
        self.max_buffer_len = max_buffer_len
        self.lock = threading.Lock()
        self.active = False
        self.freq = freq
        self.index_last = [0]
        self.state = True 

    def start_read(self):
        self.active = True
        while self.active:
            if self.freq is not None:
                time.sleep(self.freq)
            ct = time.time()
            self.buffer.append(self.state)
            self.timestamp.append(ct)
            if self.lock.acquire(blocking=False):
                if len(self.buffer) > self.max_buffer_len:
                    del self.buffer[:-self.max_buffer_len], self.timestamp[:-self.max_buffer_len]
                self.index_last[0] = len(self.buffer) - 1
                self.lock.release()
                
    def close_gripper(self):
        self.state = False 
    
    def open_gripper(self):
        self.state = True 
    
    def end_read(self):
        self.active = False 
        self.buffer = []
        self.timestamp = []

class DataSync:
    """
    A class for synchronizing data based on timestamps
    each of the object it takes in has a timestamp and a buffer attribute
    we record data at a fixed frequency
    """
    def __init__(self, datasources : list, frequency=1/30):
        self.datasources = datasources
        self.f = frequency
        self.buffers = [d.buffer for d in datasources]
        self.timestamps = [d.timestamp for d in datasources]
        self.locks = [d.lock for d in datasources]
        self.indices_last = [d.index_last for d in datasources]
        self.active = False
        self.aggregated_data = []
        self.pause_status = False
    
    def pause(self):
        self.pause_status = True
    
    def resume(self):
        self.pause_status = False
    
    def start(self):
        """
        naively take the last timestamp and record the data
        """
        self.active = True
        while self.active:
            time.sleep(self.f)
            if self.pause_status:
                continue
            [l.acquire() for l in self.locks] 
            self.aggregated_data.append([(b[:idx][-1], t[:idx][-1]) for (idx,), b, t in zip(self.indices_last, self.buffers, self.timestamps)])
            [l.release() for l in self.locks]

    def start_ros(self):
        """
        select timestamp the same way as ros does
        """
        self.active = True
        while self.active:
            time.sleep(self.f)
            if self.pause_status:
                continue
            stamp = time.time()
            [l.acquire() for l in self.locks] 
            stamps, skip_one = [], False
            for queue in self.timestamps:
                topic_stamps = []
                for s in queue:
                    stamp_delta = abs(s - stamp)
                    if stamp_delta > self.f:
                        continue  # far over the slop
                    topic_stamps.append((s, stamp_delta))
                if not topic_stamps:
                    [l.release() for l in self.locks]
                    print("skipping one")
                    skip_one = True
                    break
                topic_stamps = sorted(topic_stamps, key=lambda x: x[1])
                stamps.append(topic_stamps)
            if skip_one:
                continue
            for vv in itertools.product(*[next(iter(zip(*s))) for s in stamps]):
                vv = list(vv)
                # insert the new message
                qt = list(zip(self.buffers, self.timestamps, vv))
                if ( ((max(vv) - min(vv)) < self.f) and
                    (len([1 for b, ts, t in qt if t not in ts]) == 0) ):
                    msgs = [(b[ts.index(t)], t) for b, ts, t in qt]
                    self.aggregated_data.append(msgs)
                   
                    break  # fast finish after the synchronization
            [l.release() for l in self.locks]

    def stop(self):
        self.active = False
        pt = self.aggregated_data
        self.aggregated_data = []
        return pt

def discover_cams():
    import pyrealsense2 as rs
    """Returns a list of the ids of all cameras connected via USB."""
    ctx = rs.context()
    ctx_devs = list(ctx.query_devices())
    ids = []
    for i in range(ctx.devices.size()):
        ids.append(ctx_devs[i].get_info(rs.camera_info.serial_number))
    return ids

def take_image(fp : str):
    from perception import RgbdSensorFactory
    os.makedirs(fp, exist_ok=True)

    ids = discover_cams()
    assert ids, "[!] No camera detected."
    cfg = {}
    cfg["cam_id"] = ids[0]
    cfg["filter_depth"] = True
    cfg["frame"] = "realsense"

    sensor = RgbdSensorFactory.sensor("realsense", cfg) 
    sensor.start()

    color_im, depth_im = sensor.frames()
    np.save(os.path.join(fp, "depth_im.npy"), depth_im.data.data)
    Image.fromarray(color_im.data).save(os.path.join(fp, "color_im.png"))
