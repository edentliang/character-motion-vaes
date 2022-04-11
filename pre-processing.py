import os
from fairmotion.data import bvh, asfamc
from fairmotion.viz import bvh_visualizer
import numpy as np


mocap_datapath= './cmu-mocap-master/data/'
running_bvh = ['002','009','016','035','038']
walking_bvh = []

data = []
bad_indices = []

def pre_processing():

    for folder in ['002']:
        data_path = mocap_datapath + folder +'/'
        if folder[0] == '0':
            folder = folder[1:]
        files = os.listdir(data_path)
        i = 0
        for f in files:
            
            motion = bvh.load(data_path + f)
            positions = motion.positions(local=False)  # (frames, joints, 3)
            velocities = positions[1:] - positions[:-1]
            orientations = motion.rotations(local=False)[..., :, :2].reshape(-1, 31, 6)
            rx = velocities[:,0,0]
            ry = velocities[:,0,1]
            rz = np.arctan2(motion.rotations(local=False)[:,0,1,0],motion.rotations(local=False)[:,0,0,0])
            
            ra = rz[1:]-rz[:-1]

            sample = np.stack((rx,ry,ra),axis=-1)
            sample = np.vstack((np.zeros(3),sample))
            velocities = np.vstack(([np.zeros((31,3))],velocities))
            sample = np.append(sample,positions.reshape(positions.shape[0],-1),axis=1)
            sample = np.append(sample,velocities.reshape(velocities.shape[0],-1),axis=1)
            sample = np.append(sample,orientations.reshape(orientations.shape[0],-1),axis=1)

            data.append(sample)
            bad_indices.append(sample.shape[0])

            np.savez('mocap',data=np.array(data),end_indices=np.array(bad_indices))
            return 
    print("pre-process data done!")

        
pre_processing()


    




    

# motion = asfamc.load("02.asf","02_01.amc")
# positions = motion.positions(local=False)  # (frames, joints, 3)
# velocities = positions[1:] - positions[:-1]
# orientations = motion.rotations(local=False)[..., :, :2].reshape(-1, 31, 6)

# rx = velocities[:,0,0]
# ry = velocities[:,0,1]
# ra = np.arctan2(motion.rotations(local=False)[:,0,1,0],motion.rotations(local=False)[:,0,0,0])

# bvh.save(motion, "test.bvh")

# import scipy.ndimage.filters as filters

# import BVH as BVH
# import Animation as Animation
# from  Quaternions import  Quaternions
# from Pivots import Pivots

def process_data(anim, phase, gait, type='flat'):
    
    """ Do FK """
    global_xforms = Animation.transforms_global(anim)
    global_positions = global_xforms[:,:,:3,3] / global_xforms[:,:,3:,3]
    global_rotations = Quaternions.from_transforms(global_xforms)
    
    """ Extract Forward Direction """
    
    sdr_l, sdr_r, hip_l, hip_r = 18, 25, 2, 7
    across = (
        (global_positions[:,sdr_l] - global_positions[:,sdr_r]) + 
        (global_positions[:,hip_l] - global_positions[:,hip_r]))
    across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]
    
    """ Smooth Forward Direction """
    
    direction_filterwidth = 20
    forward = filters.gaussian_filter1d(
        np.cross(across, np.array([[0,1,0]])), direction_filterwidth, axis=0, mode='nearest')    
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]

    root_rotation = Quaternions.between(forward, 
        np.array([[0,0,1]]).repeat(len(forward), axis=0))[:,np.newaxis] 
    
    """ Local Space """
    
    local_positions = global_positions.copy()
    local_positions[:,:,0] = local_positions[:,:,0] - local_positions[:,0:1,0]
    local_positions[:,:,2] = local_positions[:,:,2] - local_positions[:,0:1,2]
    
    local_positions = root_rotation[:-1] * local_positions[:-1]
    local_velocities = root_rotation[:-1] *  (global_positions[1:] - global_positions[:-1])
    local_rotations = abs((root_rotation[:-1] * global_rotations[:-1])).log()
    
    root_velocity = root_rotation[:-1] * (global_positions[1:,0:1] - global_positions[:-1,0:1])
    root_rvelocity = Pivots.from_quaternions(root_rotation[1:] * -root_rotation[:-1]).ps
    
    """ Foot Contacts """
    
    fid_l, fid_r = np.array([4,5]), np.array([9,10])
    velfactor = np.array([0.02, 0.02])
    
    feet_l_x = (global_positions[1:,fid_l,0] - global_positions[:-1,fid_l,0])**2
    feet_l_y = (global_positions[1:,fid_l,1] - global_positions[:-1,fid_l,1])**2
    feet_l_z = (global_positions[1:,fid_l,2] - global_positions[:-1,fid_l,2])**2
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor)).astype(np.float)
    
    feet_r_x = (global_positions[1:,fid_r,0] - global_positions[:-1,fid_r,0])**2
    feet_r_y = (global_positions[1:,fid_r,1] - global_positions[:-1,fid_r,1])**2
    feet_r_z = (global_positions[1:,fid_r,2] - global_positions[:-1,fid_r,2])**2
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float)
    
    """ Phase """
    
    dphase = phase[1:] - phase[:-1]
    dphase[dphase < 0] = (1.0-phase[:-1]+phase[1:])[dphase < 0]
    
    """ Adjust Crouching Gait Value """
    
    if type == 'flat':
        crouch_low, crouch_high = 80, 130
        head = 16
        gait[:-1,3] = 1 - np.clip((global_positions[:-1,head,1] - 80) / (130 - 80), 0, 1)
        gait[-1,3] = gait[-2,3]

    """ Start Windows """
    
    Pc, Xc, Yc = [], [], []
    
    for i in range(window, len(anim)-window-1, 1):
        
        rootposs = root_rotation[i:i+1,0] * (global_positions[i-window:i+window:10,0] - global_positions[i:i+1,0])
        rootdirs = root_rotation[i:i+1,0] * forward[i-window:i+window:10]    
        rootgait = gait[i-window:i+window:10]
        
        Pc.append(phase[i])
        
        Xc.append(np.hstack([
                rootposs[:,0].ravel(), rootposs[:,2].ravel(), # Trajectory Pos
                rootdirs[:,0].ravel(), rootdirs[:,2].ravel(), # Trajectory Dir
                rootgait[:,0].ravel(), rootgait[:,1].ravel(), # Trajectory Gait
                rootgait[:,2].ravel(), rootgait[:,3].ravel(), 
                rootgait[:,4].ravel(), rootgait[:,5].ravel(), 
                local_positions[i-1].ravel(),  # Joint Pos
                local_velocities[i-1].ravel(), # Joint Vel
                ]))
        
        rootposs_next = root_rotation[i+1:i+2,0] * (global_positions[i+1:i+window+1:10,0] - global_positions[i+1:i+2,0])
        rootdirs_next = root_rotation[i+1:i+2,0] * forward[i+1:i+window+1:10]   
        
        Yc.append(np.hstack([
                root_velocity[i,0,0].ravel(), # Root Vel X
                root_velocity[i,0,2].ravel(), # Root Vel Y
                root_rvelocity[i].ravel(),    # Root Rot Vel
                dphase[i],                    # Change in Phase
                np.concatenate([feet_l[i], feet_r[i]], axis=-1), # Contacts
                rootposs_next[:,0].ravel(), rootposs_next[:,2].ravel(), # Next Trajectory Pos
                rootdirs_next[:,0].ravel(), rootdirs_next[:,2].ravel(), # Next Trajectory Dir
                local_positions[i].ravel(),  # Joint Pos
                local_velocities[i].ravel(), # Joint Vel
                local_rotations[i].ravel()   # Joint Rot
                ]))
                                                
    return np.array(Pc), np.array(Xc), np.array(Yc)
 