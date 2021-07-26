import argparse
import os
import random
import shutil
import sys

import numba
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

sys.path.append(os.getcwd())

__all__ = ['evaluate']

_configs = None 
model = None
model_loaded = False

def Pointscallback(msg):
    global model
    pc = ros_numpy.numpify(msg)
    points = np.zeros((pc.shape[0],4))
    points[:,0]=pc['x']
    points[:,1]=pc['y']
    points[:,2]=pc['z']
    points[:,3]=pc['intensity']
    # print(self.anchor_points.shape)
    # p = pcl.PointCloud(np.array(points, dtype=np.float32))

    global model_loaded, _configs
    if model_loaded is not True:
        return  

    # predictions = np.zeros((len(dataset), 8))
    size_templates = _configs.data.size_templates.to(_configs.device)
    heading_angle_bin_centers = torch.arange(
        0, 2 * np.pi, 2 * np.pi / _configs.data.num_heading_angle_bins).to(_configs.device)
    current_step = 0

    points = np.reshape(points, (64, 4, -1))
    points = points.astype(np.float32)    
    inputs = dict()
    inputs['features'] = torch.from_numpy(points)
    # print(inputs.shape)
    one_hot_vectors = np.zeros(_configs.model.num_classes)
    for i in range(points.shape[0]):
        if(len(one_hot_vectors) >= points.shape[0]*3):
            break
        one_hot_vectors = np.append(one_hot_vectors, np.array([1,0,0]), axis = 0)
    one_hot_vectors = np.reshape(one_hot_vectors, (-1, 3))
    one_hot_vectors = one_hot_vectors.astype(np.float32)    

    print(one_hot_vectors.shape)

    inputs['one_hot_vectors'] = torch.from_numpy(one_hot_vectors)
    print(inputs['one_hot_vectors'].dim())
    # inputs['one_hot_vector'] = one_hot_vector[self.class_name_to_class_id[class_name]] = 1
    with torch.no_grad():  
    #     for k, v in inputs.items():
    #         # print(len(k), len(v))
        inputs['features'] = inputs['features'].to(_configs.device, non_blocking=True)
        outputs = model(inputs)

        center = outputs['center']  # (B, 3)
        heading_scores = outputs['heading_scores']  # (B, NH)
        heading_residuals = outputs['heading_residuals']  # (B, NH)
        size_scores = outputs['size_scores']  # (B, NS)
        size_residuals = outputs['size_residuals']  # (B, NS, 3)

        batch_size = center.size(0)
        batch_id = torch.arange(batch_size, device=center.device)
        heading_bin_id = torch.argmax(heading_scores, dim=1)
        heading = heading_angle_bin_centers[heading_bin_id] + heading_residuals[batch_id, heading_bin_id]  # (B, )
        size_template_id = torch.argmax(size_scores, dim=1)
        size = size_templates[size_template_id] + size_residuals[batch_id, size_template_id]  # (B, 3)

        center = center.cpu().numpy()
        heading = heading.cpu().numpy()
        size = size.cpu().numpy()

    print("center: ", center)
    # print("heading_scores: ", heading_scores)
    # print("heading_residuals: ", heading_residuals)
    # print("size_scores: ", size_scores)
    # print("size_residuals: ", size_residuals)

        # rotation_angle = targets['rotation_angle'].cpu().numpy()  # (B, )
        # rgb_score = targets['rgb_score'].cpu().numpy()  # (B, )

        # update_predictions(predictions=predictions, center=center, heading=heading, size=size,
        #                     rotation_angle=rotation_angle, rgb_score=rgb_score,
        #                     current_step=current_step, batch_size=batch_size)
        # current_step += batch_size

    # np.save(configs.evaluate.stats_path, predictions)
    # image_ids = write_predictions(configs.evaluate.predictions_path, ids=dataset.data.ids,
    #                                 classes=dataset.data.class_names, boxes_2d=dataset.data.boxes_2d,
    #                                 predictions=predictions, image_id_file_path=configs.evaluate.image_id_file_path)
    # _, current_results = eval_from_files(prediction_folder=configs.evaluate.predictions_path,
    #                                         ground_truth_folder=configs.evaluate.ground_truth_path,
    #                                         image_ids=image_ids, verbose=True)
    # if configs.evaluate.num_tests == 1:
    #     return
    # else:
    #     for class_name, v in current_results.items():
    #         if class_name not in results:
    #             results[class_name] = dict()
    #         for kind, r in v.items():
    #             if kind not in results[class_name]:
    #                 results[class_name][kind] = []
    #             results[class_name][kind].append(r)

    # for class_name, v in results.items():
    #     print(f'{class_name}  AP(Average Precision)')
    #     for kind, r in v.items():
    #         r = np.asarray(r)
    #         m = r.mean(axis=0)
    #         s = r.std(axis=0)
    #         u = r.max(axis=0)
    #         rs = ', '.join(f'{mv:.2f} +/- {sv:.2f} ({uv:.2f})' for mv, sv, uv in zip(m, s, u))
    #         print(f'{kind:<4} AP: {rs}')


def prepare():
    from utils.common import get_save_path
    from utils.config import configs
    from utils.device import set_cuda_visible_devices

    # since PyTorch jams device selection, we have to parse args before import torch (issue #26790)
    parser = argparse.ArgumentParser()
    parser.add_argument('configs', nargs='+')
    parser.add_argument('--devices', default=None)
    args, opts = parser.parse_known_args()
    if args.devices is not None and args.devices != 'cpu':
        gpus = set_cuda_visible_devices(args.devices)
    else:
        gpus = []

    print(f'==> loading configs from {args.configs}')
    configs.update_from_modules(*args.configs)
    # define save path
    save_path = get_save_path(*args.configs, prefix='runs')
    os.makedirs(save_path, exist_ok=True)
    configs.train.save_path = save_path
    configs.train.best_checkpoint_path = os.path.join(save_path, 'best.pth.tar')

    # override configs with args
    configs.update_from_arguments(*opts)
    if len(gpus) == 0:
        configs.device = 'cpu'
        configs.device_ids = []
    else:
        configs.device = 'cuda'
        configs.device_ids = gpus

    if 'best_checkpoint_path' not in configs.rosinference or configs.rosinference.best_checkpoint_path is None:
        if 'best_checkpoint_path' in configs.train and configs.train.best_checkpoint_path is not None:
            configs.rosinference.best_checkpoint_path = configs.train.best_checkpoint_path
        else:
            configs.rosinference.best_checkpoint_path = os.path.join(configs.train.save_path, 'best.pth.tar')
    assert configs.rosinference.best_checkpoint_path.endswith('.pth.tar')

    return configs


def rosinference(configs=None):
    rospy.init_node('PointNet_inference', anonymous=True)
    sub_points = rospy.Subscriber("/os_cloud_node/below_points", PointCloud2, Pointscallback, queue_size = 1)
    configs = prepare() if configs is None else configs

    global model

    import time

    from tqdm import tqdm

    from ..utils import eval_from_files

    ###########
    # Prepare #
    ###########

    if configs.device == 'cuda':
        cudnn.benchmark = True
        if configs.get('deterministic', False):
            cudnn.deterministic = True
            cudnn.benchmark = False
    if ('seed' not in configs) or (configs.seed is None):
        configs.seed = torch.initial_seed() % (2 ** 32 - 1)

    #################################
    # Initialize Model #
    #################################

    print(f'\n==> creating model "{configs.model}"')
    model = configs.model()
    if configs.device == 'cuda':
        model = torch.nn.DataParallel(model)
    model = model.to(configs.device)

    if os.path.exists(configs.rosinference.best_checkpoint_path):
        print(f'==> loading checkpoint "{configs.rosinference.best_checkpoint_path}"')
        checkpoint = torch.load(configs.rosinference.best_checkpoint_path)
        model.load_state_dict(checkpoint.pop('model'))
        del checkpoint
    else:
        return

    model.eval()

    global model_loaded, _configs
    model_loaded = True
    _configs = configs

    print("MODEL IS LOADED")

    ##############
    # Evaluation #
    ##############
    print("********" *6)
    print("Model inference with ROS message have started")

    try:
        rospy.spin()        
    except KeyboardInterrupt():
        print("ROS terminated")
    

@numba.jit()
def update_predictions(predictions, center, heading, size, rotation_angle, rgb_score, current_step, batch_size):
    for b in range(batch_size):
        l, w, h = size[b]
        x, y, z = center[b]  # (3)
        r = rotation_angle[b]
        t = heading[b]
        s = rgb_score[b]
        v_cos = np.cos(r)
        v_sin = np.sin(r)
        cx = v_cos * x + v_sin * z  # it should be v_cos * x - v_sin * z, but the rotation angle = -r
        cy = y + h / 2.0
        cz = v_cos * z - v_sin * x  # it should be v_sin * x + v_cos * z, but the rotation angle = -r
        r = r + t
        while r > np.pi:
            r = r - 2 * np.pi
        while r < -np.pi:
            r = r + 2 * np.pi
        predictions[current_step + b] = [h, w, l, cx, cy, cz, r, s]


def write_predictions(prediction_path, ids, classes, boxes_2d, predictions, image_id_file_path=None):
    import pathlib

    # map from idx to list of strings, each string is a line (with \n)
    results = {}
    for i in range(predictions.shape[0]):
        idx = ids[i]
        output_str = ('{} -1 -1 -10 '
                      '{:f} {:f} {:f} {:f} '
                      '{:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f}\n'.format(classes[i], *boxes_2d[i][:4], *predictions[i]))
        if idx not in results:
            results[idx] = []
        results[idx].append(output_str)

    # write txt files
    if os.path.exists(prediction_path):
        shutil.rmtree(prediction_path)
    os.mkdir(prediction_path)
    for k, v in results.items():
        file_path = os.path.join(prediction_path, f'{k:06d}.txt')
        with open(file_path, 'w') as f:
            f.writelines(v)

    if image_id_file_path is not None and os.path.exists(image_id_file_path):
        with open(image_id_file_path, 'r') as f:
            val_ids = f.readlines()
        for idx in val_ids:
            idx = idx.strip()
            file_path = os.path.join(prediction_path, f'{idx}.txt')
            if not os.path.exists(file_path):
                # print(f'warning: {file_path} doesn\'t exist as indicated in {image_id_file_path}')
                pathlib.Path(file_path).touch()
        return image_id_file_path
    else:
        image_ids = sorted([k for k in results.keys()])
        return image_ids


if __name__ == '__main__':
    rosinference()
