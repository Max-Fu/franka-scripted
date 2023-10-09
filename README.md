# Running Automatic Cube Data Collection on Franka
This repository shows how to run automatic cube data collection on a Franka Emika Robot. The setup requires the franka connected to a NUC (running realtime system) via ethernet, a workstation connected to the NUC via ethernet. 3 Logitech Brio cameras are connected to the workstation. The NUC's IP is set to `10.0.0.1` and we set the workstation's IP to `10.0.0.2`.

## Polymetis installation 
### On NUC: 

First open firefox to enable fcl on franka:
```
ssh -X <username>@10.0.0.1
firefox
https://172.16.0.2/desk/
```

Then, create conda environment and activate it.
```
ssh <username>@10.0.0.1
conda create -n polymetis python=3.8
conda activate polymetis
conda install mamba -n base -c conda-forge
mamba install -c pytorch -c fair-robotics -c aihabitat -c conda-forge polymetis
```

Ideally in a tmux session, fire up the robot server 
```
sudo pkill -9 run_server
launch_robot.py robot_client=franka_hardware
```

In another tmux session, fire up the gripper server 
```
launch_gripper.py gripper=franka_hand
```

### On the workstation 
Install polymetis, note that the dependency is different from this repo. 
```
conda create -n polymetis python=3.8
conda activate polymetis
conda install mamba -n base -c conda-forge
mamba install pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
mamba install -c pytorch -c fair-robotics -c aihabitat -c conda-forge polymetis
```
Clone fairo 
```
git clone https://github.com/facebookresearch/fairo.git
cd fairo/polymetis/examples
```

Since the ip address of NUC is `10.0.0.1`, we have to change the example a bit: 
For each of 
```
robot = RobotInterface(
    ip_address="localhost",
)
```
change it to 
```
robot = RobotInterface(
    ip_address="10.0.0.1",
)
```

On Franka Desk / Setting, update mass and inertia matrix according to 
```
https://github.com/frankaemika/external_gripper_example/blob/master/panda_with_robotiq_gripper_example/config/endeffector-config.json
```
This will compensate for the additional weight of the camera on the end effector. 

Note that franka end effector quaternion uses xyzw formulation, and the "standard" axes are the ones provided by Franka, which happen to be visually off by [45 degrees](https://github.com/facebookresearch/fairo/issues/1223) from the symmetries of the Franka Hand shape. 


### Set up camera 
Run `v4l2-ctl --list-devices`. 
To test out particular camera feed: first install `sudo apt install ffmpeg`, then based on `v4l2` run `ffplay /dev/video{i}` 

Using this, find out the camera id correspond to left, hand, and right camera (just the numbers). 

To list the available controls: `v4l2-ctl -d /dev/video8 -l` 

Current setting (all three cameras): 
```
v4l2-ctl -d /dev/video0 -c focus_auto=0
v4l2-ctl -d /dev/video4 -c focus_auto=0
v4l2-ctl -d /dev/video8 -c focus_auto=0
```

### Installation
```
pip install -r requirements.txt
```

To install `fcl`, first, install octomap, which is necessary to use OcTree. For Ubuntu, use `sudo apt-get install liboctomap-dev`. Second, install FCL using the instructions provided here. If you're on Ubuntu 17.04 or newer, you can install FCL using `sudo apt-get install libfcl-dev`. 

Then we install the Python wrappers for FCL:
```
pip install python-fcl
```

Install Python packages from `third_party`:

```
pushd third_party/autolab_core && pip install -e . && popd
pushd third_party/perception && pip install -e . && popd
```

# Data Collection 
You can remove or add `--gripper_blocking` to record or not record data while closing gripper. Additional flags include `--data_dir`, `--dir_name`, `--num_demos`, `--wrist_cam_id`, `--vis`. By default, `wrist_cam_id=(4,0,8)`, which represent the left camera, hand camera, and right camera.
## Data collection for Picking 1/n objects
```
python tools/gen_structured_grasp.py --num_demos 40 --dir_name pick-green-cube
```

## Data Collection for Stacking 1/n objects
```
python tools/gen_structured_stack.py --num_demos 40 --dir_name stack-green-yellow-cube
```

## (Experimental) Data Collection for DexNet 
(Warning) This requires to build polymetis from scratch, which involves modification to polymetis' setup. The conda environment name will be `polymetis-local`, adjust accordingly if needed in `cfg/environment.yml`. Disclaimer: since the depth camera has a lower resolution and precision and we are using a different setup than the original DexNet setup, the grasps are not guaranteed to be as precise as the original DexNet formulation. 

### Installation
Please reference the official polymetis build [documentation](https://facebookresearch.github.io/fairo/polymetis/installation.html#for-advanced-users-developers). 
```
# Clone fairo
git clone git@github.com:facebookresearch/fairo
cd fairo/polymetis

# replace the environment config file 
mv polymetis/environment.yml polymetis/environment_old.yml
cp <franka-scripted path>/cfg/environment.yml polymetis/environment.yml

# one system level installation
sudo apt-get install libboost-all-dev

# create conda environment
conda env create -f ./polymetis/environment.yml
conda activate polymetis-local
conda install mamba -n base -c conda-forge
mamba install -y -c conda-forge poco
mamba install -y boost
pip install -e ./polymetis
pip install pyglet==1.4.10
mamba install -y tensorflow-gpu=1.14

# compile polymetis 
mkdir -p ./polymetis/build
cd ./polymetis/build

cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_FRANKA=OFF -DBUILD_TESTS=OFF -DBUILD_DOCS=OFF
make -j

# Installing DexNet
pushd third_party/autolab_core && pip install -e . && popd
pushd third_party/perception && pip install -e . && popd
pushd third_party/gqcnn && pip install -e . && popd
```

Common failures:
1. gRPC related errors (i.e. CMake Error at grpc/cmake/build/gRPCConfig.cmake:15 (include): ...)

Solution: `mamba install -c conda-forge grpc-cpp==1.41.1`

2. Boost related errors (i.e. Could NOT find Boost (missing: serialization))

Solution: `sudo apt-get install libboost-all-dev`

### Calibration
In our experiments, we use an overhead realsense L515 camera. To run the DexNet code base, we require a camera extrinsics calibration file. An example is provided in `cfg/T_realsense_world.tf`. The format follows the definition in [autolab_core](https://github.com/BerkeleyAutomation/autolab_core/blob/04a75b55d1e8cf51c21a9f84f9ce813cf840351a/autolab_core/rigid_transformations.py#L544):
```
realsense
world
translation (space separated)
rotation_row_0 (space separated)
rotation_row_1 (space separated)
rotation_row_2 (space separated)
```
We use the CV2 [checkerboard](https://github.com/opencv/opencv/blob/4.x/doc/pattern.png) to obtain the camera calibration. We set the checkerboard to have solid square corners in the robots positive y direction. A procedure that we use is to first find the chessboard in robot transformation by manually placing the robot's end effector on the chessboard to obtain `cfg/T_gripper_world.tf`, then move the end effector away so that the chessboard is in the camera's view. We then move the end effector away from the checkerboard, and run 
```
python tools/camera_registration.py
```
This will save the camera pose in robot base frame to `cfg/T_realsense_world.tf`.

### Data Collection
To start collecting data, run 
```
python tools/gen_demos_grasp.py --num_demos 40 --dir_name dexnet_bin_pick
```
