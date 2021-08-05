import argparse
import math
import os
from datetime import datetime

import h5py
import numpy as np
import plyfile
from matplotlib import cm

import rospy
import rospkg
import ros_numpy
import math
import sys
import cv2
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
import sensor_msgs.point_cloud2 as pc2
import pcl
from pcl_helper import *
from std_msgs.msg import Header


class datasetVis(object):

    def __init__(self):
        rospy.init_node('tutorial', anonymous=True) 
        self.rate = rospy.Rate(10)
        self.pub = rospy.Publisher("/kitti_dataset", PointCloud2, queue_size=1)


        default_data_dir = '/home/usrg/Dataset/LidarDetection/niro'
        default_output_dir = 'data/s3dis/pointcnn'
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--data', dest='data_dir', default=default_data_dir,
                            help=f'Path to S3DIS data (default is {default_data_dir})')
        parser.add_argument('-f', '--folder', dest='output_dir', default=default_output_dir,
                            help=f'Folder to write labels (default is {default_output_dir})')
        parser.add_argument('--max_num_points', '-m', help='Max point number of each sample', type=int, default=230400)
        parser.add_argument('--block_size', '-b', help='Block size', type=float, default=1.5)
        parser.add_argument('--grid_size', '-g', help='Grid size', type=float, default=0.03)
        parser.add_argument('--save_ply', '-s', help='Convert .pts to .ply', action='store_true')

        args = parser.parse_args()

        self.visualize(data_dir=args.data_dir, output_dir=args.output_dir)


    def visualize(self, data_dir, output_dir):
        object_dict = {
            'Pedestrian': 0,
            'Car': 1,
            'Cyclist': 2,
            'Van': 1,
            'Truck': -3,
            'Person_sitting': 0,
            'Tram': -99,
            'Misc': -99,
            'DontCare': -1
        }

        training_path = os.path.join(data_dir, 'training')
        if not os.path.isdir(training_path):
            print("No such directory.")
            return

        # pc_path_dir = os.path.join(training_path, 'velodyne') # kitti
        # label_path_dir = os.path.join(training_path, 'label_2') # kitti

        pc_path_dir = os.path.join(training_path, 'lidar') #niro
        label_path_dir = os.path.join(training_path, 'label') # niro

        pc_dir = os.listdir(pc_path_dir)
        pc_dir = sorted(pc_dir)
        # label_objects = os.listdir(label_path_dir)
        print(len(pc_dir))

        xyz_room = np.zeros((1, 6))
        label_room = np.zeros((1, 1))
    

        for pc_buf in pc_dir:
            pc_ind = pc_buf.split('.', 1)[0]
            # pc_path = os.path.join(pc_path_dir, pc_ind + '.bin') # kitti
            pc_path = os.path.join(pc_path_dir, pc_ind + '.npy') # niro
            label_path = os.path.join(label_path_dir, pc_ind + '.txt')

            print(pc_path, label_path)
            
            # xyzi_arr = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4) #kitti
            xyzi_arr = np.load(pc_path).reshape(-1, 4) #niro
            xyzi_arr = np.array(xyzi_arr[:, :3])



            # ori_pc = pcl.PointCloud(xyzi_arr)
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'velodyne'

            # fields = [PointField('x', 0, PointField.FLOAT32, 1),
            #         PointField('y', 4, PointField.FLOAT32, 1),
            #         PointField('z', 8, PointField.FLOAT32, 1),        
            #         PointField('intensity', 12, PointField.FLOAT32, 1)]  

            # ros_msg = pc2.create_cloud(header, fields, xyzi_arr)
            
            ros_msg = pc2.create_cloud_xyz32(header, xyzi_arr)


            num_channel = int(xyzi_arr.shape[0] / 64)
            print('# of channel : ', num_channel, ', shape: ', xyzi_arr.shape)


            self.pub.publish(ros_msg)

            self.rate.sleep()


def main():
    vis = datasetVis()

if __name__ == '__main__':
    main()
