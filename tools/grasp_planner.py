# Authors: Justin Kerr and Max Fu
# Originated from Evo-NeRF: Evolving NeRF for Sequential Robot Grasping of Transparent Objects
# https://openreview.net/forum?id=Bxr45keYrf

from queue import Empty
try:
    from gqcnn.grasping.policy import FullyConvolutionalGraspingPolicyParallelJaw
    from gqcnn.grasping import RgbdImageState
except:
    print("GQCNN or tensorflow not installed, dex-net grasp planner will be unavailable")
from autolab_core import YamlConfig,DepthImage, RigidTransform, CameraIntrinsics,RgbdImage,ColorImage,BinaryImage,Box,Point
import numpy as np
from visualization import Visualizer2D as v2d
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R
import multiprocessing as mp
import cv2 
from sklearn.cluster import KMeans

def gui_masking(dimg:DepthImage)->np.ndarray:
    '''
    Spins up a gui for selecting a crop of the image, and returns the mask corresponding to the crop
    '''
    fig,ax = plt.subplots(1)
    ax.imshow(dimg.data)
    xlims,ylims=None,None
    def on_xlims_change(event_ax):
        nonlocal xlims
        xlims = event_ax.get_xlim()

    def on_ylims_change(event_ax):
        nonlocal ylims
        ylims = event_ax.get_ylim()

    ax.callbacks.connect('xlim_changed', on_xlims_change)
    ax.callbacks.connect('ylim_changed', on_ylims_change)
    plt.show()
    mask = np.zeros_like(dimg.data,dtype=np.uint8)
    if xlims is None:
        return mask
    mask[int(ylims[1]):int(ylims[0]),int(xlims[0]):int(xlims[1])] = 255
    return mask

def box_masking(dimg:DepthImage, camera_pose:RigidTransform, camera_intr:CameraIntrinsics,
                    world_box:Box=Box(np.array((.1,-.4,.02)),np.array((.6,.4,.5)),'base_link'))->np.ndarray:
    '''
    Masks the depth image based on the Box given

    dimg: depth image in camera frame
    world_box: autolab_core Box object representing the bounds of the depth map to keep in WORLD coordinates
    camera_pose: camera pose in world frame
    camera_intr: camera intrinsics
    '''
    pc = camera_pose*camera_intr.deproject(dimg)
    pc,_=pc.box_mask(world_box)
    masked_depth = camera_intr.project_to_image(camera_pose.inverse()*pc)
    mask = (masked_depth.data!=0).astype(np.uint8)
    mask[mask>0] = 255
    return mask

class AsyncDexNet(mp.Process):
    def __init__(self,conf):
        super().__init__()
        self.depth_q = mp.Manager().Queue()#queue which holds depth inputs
        self.grasp_q = mp.Manager().Queue()#queue which returns grasps
        self.daemon=True
        self.conf=conf
        self.start()

    def run(self):
        #os.environ['CUDA_VISIBLE_DEVICES']='0'
        self.fc_sampler=FullyConvolutionalGraspingPolicyParallelJaw(self.conf)
        while True:
            try:
                eval_args = self.depth_q.get()
            except Empty:
                continue
            grasps = self.fc_sampler._action(*eval_args)
            self.grasp_q.put(grasps)

    def _action(self,imstate,numgrasps):
        self.depth_q.put((imstate,numgrasps))
        return self.grasp_q.get()


class DexNetGraspPlanner:
    def __init__(self, intr:CameraIntrinsics=CameraIntrinsics.load('cfg/realsense.intr'), visualize=False):
        conf=YamlConfig('cfg/fc_gqcnn_pj.yaml')
        self.fc_sampler=AsyncDexNet(conf['policy'])
        import time
        time.sleep(3)
        self.vis=visualize
        self.im_width=conf['policy']['metric']['fully_conv_gqcnn_config']['im_width']
        self.im_height=conf['policy']['metric']['fully_conv_gqcnn_config']['im_height']
        self.intr=intr.resize(self.im_width/(intr.width))
        #prewarm the network by doing a junk execution
        self._plan_from_dimg(DepthImage(np.zeros((self.im_height,self.im_width))),np.ones((self.im_height,self.im_width),dtype=np.uint8),warm=True)
    
    def plan_from_dimg(self, dimg: np.ndarray, camera_pose: RigidTransform, create_mask=True):#(h,w) image
        dimg=cv2.resize(dimg, (self.im_width, self.im_height))
        dimg=DepthImage(dimg[...,0],frame=self.intr.frame)
        if create_mask:
            if os.path.exists("cfg/mask.npy"):
                mask = np.load("cfg/mask.npy")
            else:
                mask = gui_masking(dimg)
                np.save("cfg/mask.npy", mask)
            grasps = self._plan_from_dimg(dimg, mask)
        else:
            grasps = self._plan_from_dimg(dimg)
        best_grasp = grasps[0]
        return self.get_grasp_tf(dimg, best_grasp.center, best_grasp.angle, camera_pose)

    def plan_stack_from_dimg(self, dimg: np.ndarray, camera_pose: RigidTransform, create_mask=True, delta=0.01):#(h,w) image
        """
        Plan a stacking grasp pose from a depth image 
        returns two grasps transforms: one for grasping and another for stacking
        delta is the tolerance away from the min point on the masked depth image that is accepted for grasping
        """
        dimg=cv2.resize(dimg, (self.im_width, self.im_height))
        dimg=DepthImage(dimg[...,0],frame=self.intr.frame)
        if create_mask:
            if os.path.exists("cfg/mask.npy"):
                mask = np.load("cfg/mask.npy")
            else:
                mask = gui_masking(dimg)
                np.save("cfg/mask.npy", mask)
            grasps = self._plan_from_dimg(dimg, mask)
        else:
            grasps = self._plan_from_dimg(dimg)
        
        # now perform k means clustering based on the centers of the grasps
        mask = mask.astype(np.bool)
        # apply depth mask and find segmentation
        min_depth = np.min(dimg.data[mask])
        apply_mask = dimg.data * mask
        coarse_mask = (apply_mask < min_depth + delta) & (apply_mask > 0)
        plt.imshow(coarse_mask); plt.show()
        selected_pixels = np.argwhere(coarse_mask)[:, ::-1]
        kmeans = KMeans(n_clusters=2, random_state=0).fit(selected_pixels)

        centers = np.array([g.center.data for g in grasps])
        pred_clusters = kmeans.predict(centers)
        best_grasp, best_center_id = grasps[0], pred_clusters[0]

        # find the best grasp for the other object
        other_cube_grasps_ids = np.where(pred_clusters != best_center_id)[0]
        if len(other_cube_grasps_ids) == 0:
            cluster_centers = kmeans.cluster_centers_
            other_center = cluster_centers[1 - best_center_id]
            other_center = Point(np.array(other_center).astype(int))
            other_angle = best_grasp.angle
        else:
            other_cube_grasp_id = np.sort(other_cube_grasps_ids)[0] # note grasps are sorted with reverse q value
            other_cube_grasp = grasps[other_cube_grasp_id]
            other_center = other_cube_grasp.center
            other_angle = other_cube_grasp.angle
        return self.get_grasp_tf(dimg, best_grasp.center, best_grasp.angle, camera_pose), self.get_grasp_tf(dimg, other_center, other_angle, camera_pose, offset_z=-0.05)

    def _plan_from_dimg(self, dimg:DepthImage, grasp_mask:np.ndarray = None,save_fig=None,warm=False):
        '''
        plan a grasp using GQCNN from a depth image
        dimg: depth image
        grasp_mask: binary image with shape (im_height,im_width) with 1s where the grasp should be sampled from
        
        returns ordered list of Grasp2D objects sorted in descending order of quality
        '''
        if grasp_mask is None:
            grasp_mask = 255*np.ones((dimg.height,dimg.width),dtype=np.uint8)
        mask  = BinaryImage(grasp_mask.astype(np.uint8), frame=dimg.frame)
        im=ColorImage(np.zeros((dimg.height,dimg.width,3),dtype=np.uint8),frame=dimg.frame)
        rgbdimg = RgbdImage.from_color_and_depth(im,dimg)
        imgstate = RgbdImageState(rgbdimg,self.intr,segmask=mask)
        grasps=self.fc_sampler._action(imgstate,20)
        #sort the GraspActions by q value
        grasps.sort(key=lambda g:g.q_value,reverse=True)
        if (self.vis or save_fig) and not warm:
            vis=v2d()
            vis.imshow(dimg)
            for grasp in grasps[1:]:
                vis.grasp(grasp.grasp,scale=1.0,show_center=True,show_axis=True,color='b')
            vis.grasp(grasps[0].grasp,scale=1.0,show_center=True,show_axis=True,color='g')
            vis.title("Sampled grasps")
            if save_fig is not None:
                vis.savefig(save_fig)
                vis.clf()
            if self.vis:
                vis.show()
        #then return the Grasp2Ds correspond to the GraspActions
        return [grasp.grasp for grasp in grasps]

    def get_grasp_tf(self, img : np.ndarray, point : Point, angle : float, camera_pose : RigidTransform, window = 1, offset_z = 0.01):
        """
        return the grasp rigid transform 
        img: depth image 
        Point: 2D point
        angle: grasp angle in rad
        camera_pose: T_camera_world
        """
        # #convert to Grasp3D representation
        point._frame = self.intr.frame
        point = self.intr.deproject_pixel(np.min(img[point[1]-window:point[1]+window, point[0]-window:point[0]+window]), point)
        point_world = camera_pose*point
        point_world.data[2] -= offset_z
        y_axis_cam = R.from_euler('z', angle - np.pi / 2, degrees=False).as_matrix()
        y_axis_room = camera_pose.rotation @ y_axis_cam
        return RigidTransform(rotation=y_axis_room, translation=point_world.data, from_frame="gripper", to_frame="world")
