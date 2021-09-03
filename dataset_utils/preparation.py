'''
The following steps are required to create a dataset for training a field homography model:

1. Generate requests (game, frame id and manual PoI*) for further processing
2. Calculate homography (theta) from manually annotated PoI of a field
3. Project the field PoI to the frame PoI using the homography
4. Calculate the Reprojection RMSE** based on the reprojected and manual PoI
5. Create a segmentation mask from a field template and the homography
6. Create debug images (optional)
7. Save the results: mask, theta and PoI (and debug images)
8. Generate onehote masks from the rgb masks

* PoI - Points of Interest
** RMSE - Root Mean Square Error
'''

import os
import json
import numpy as np
import cv2

from multiprocessing import Process, Queue, cpu_count
from queue import Empty


FOOTBALL_PITCH_IGNORE_POINTS = [12, 13, 16, 19, 20]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def generate_requests(anno_dir : str):
    '''
    Generates requests (game, frame id and manual PoI) from annotations for further processing

    :param anno_dir: directory containing annotations
    :return:         generated requests
    '''
    requests = {}

    # Get all sub-directories (game names) in the anno directory:
    names = [name for name in os.listdir(anno_dir) if os.path.isdir(os.path.join(anno_dir, name))]

    for name in names:
        # Open anno:
        game_anno_path = os.path.join(anno_dir, name, 'manual_anno.json')
        with open(game_anno_path, 'r') as f:
            game_anno = json.load(f)

        game_requests = {}
        for frame_id, values in game_anno.items():
            # if values['theta'] is not None:
            game_requests[frame_id] = {
                'manual_poi': np.array(values['poi']),
                'poi': None,
                'theta': None,
                'rmse': values['rmse'] if 'rmse' in values else None
            }

        requests[name] = game_requests

    return requests


def calculate_homography(field_poi : np.array, manual_poi : np.array):
    '''
    Calculate homography (theta) from field_poi (field PoI) and manual_poi (manually annotated PoI)

    :param field_poi:  field PoI
    :param manual_poi: manually annotated PoI
    :return:           theta
    '''
    assert field_poi.shape[0] == manual_poi.shape[0]

    pts_from, pts_to = [], []
    for i, (x, y) in enumerate(manual_poi):
        if x != -1.0 and y != -1.0:
            pts_from.append(field_poi[i])
            pts_to.append(manual_poi[i])

    if len(pts_from) < 4:
        # print ('Not enough points to calculate the homography!')
        return None

    # Find the homography:
    theta, r = cv2.findHomography(np.array(pts_from), np.array(pts_to))

    return theta


def find_nonzero_points(poi, ignore_pts=None):
    if ignore_pts is None:
        ignore_pts = []

    nonzero = np.ones(poi.shape[0], dtype=bool)
    for i, (x, y) in enumerate(poi):
        if i in ignore_pts or x == -1.0 and y == -1.0:
            nonzero[i] = False

    return nonzero


def calculate_reprojection_rmse(pts1, pts2, nonzero=None, norm_size=None):
    '''
    Calculate the distance between the points
    '''
    p1 = np.copy(pts1)
    p2 = np.copy(pts2)
    if norm_size is not None:
        p1[:, 0] *= norm_size[0]
        p1[:, 1] *= norm_size[1]
        p2[:, 0] *= norm_size[0]
        p2[:, 1] *= norm_size[1]

    if nonzero is None:
        nonzero = np.ones(p1.shape[0], dtype=bool)

    dist = np.sqrt(np.sum(np.power(p1 - p2, 2), axis=1))
    num_nonzero = np.count_nonzero(nonzero, axis=0)
    rmse = np.sum(dist * nonzero, axis=0) / num_nonzero

    return rmse


def rescale_theta(src_size, dst_size, theta):
    ''' Rescales theta (homography) to the target size '''
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    src_scale = np.array([[dst_w, 0, 0], [0, dst_h, 0], [0, 0, 1]], dtype=np.float64)
    dst_scale_inv = np.array([[1 / src_w, 0, 0], [0, 1 / src_h, 0], [0, 0, 1]], dtype=np.float64)
    scaled_theta = np.matmul(np.matmul(src_scale, theta), dst_scale_inv)

    return scaled_theta


def convert_rgb_to_onehot(mask_dir, mapping):
    '''
    :mapping: Dictionary containing <id, color> paris, where color=(r,g,b)
    '''
    counter = 0

    for dirpath, dirnames, filenames in os.walk(mask_dir):
        for filename in [f for f in filenames if f.endswith('.png')]:
            mask_path = os.path.join(dirpath, filename)
            mask = cv2.imread(mask_path, 1)

            # Mapping:
            for id, color in mapping.items():
                mask[np.all(mask == color, axis=2)] = (id, 0, 0)
            mask = mask[:,:,0]

            cv2.imwrite(mask_path, mask)
            counter += 1

    print ('Done! Processed masks:',counter)


def apply_mapping_worker(worker_id, paths_queue, mapping):
    '''
    Multithreading worker
    '''
    # print ('Worker:{} started'.format(worker_id))

    while True:
        try:
            path = paths_queue.get(timeout=1)

            # Load image:
            mask = cv2.imread(path, 1)

            # Mapping:
            for id, color in mapping.items():
                mask[np.all(mask == color, axis=2)] = (id, 0, 0)
            mask = mask[:, :, 0]

            # Save the result:
            cv2.imwrite(path, mask)

        except Empty:
            break

    # print ('Worker:{} finished'.format(worker_id))


def convert_rgb_to_onehot_parallel(mask_dir, mapping, num_threads=None):
    '''
    :mapping: Dictionary containing <id, color> paris, where color=(r,g,b)
    '''
    paths_queue = Queue()
    counter = 0
    if num_threads is None:
        num_threads = cpu_count()

    # Get paths:
    for dirpath, dirnames, filenames in os.walk(mask_dir):
        for filename in [f for f in filenames if f.endswith('.png')]:
            paths_queue.put(os.path.join(dirpath, filename))
            counter += 1

    # Run workers:
    workers = []
    for i in range(num_threads):
        workers.append(Process(target=apply_mapping_worker, args=(i+1,paths_queue,mapping,)))
        workers[-1].start()

    for w in workers:
        w.join()

    print ('Done! Processed masks:',counter)


def generate_onehot(mask_dir, num_classes=8):
    mapping = {}
    if num_classes == 4:
        mapping[1] = (0, 255, 0)
        mapping[2] = (255, 0, 0)
        mapping[3] = (0, 0, 255)
    elif num_classes == 7:
        mapping[1] = (0, 255, 0)
        mapping[2] = (255, 0, 0)
        mapping[3] = (0, 0, 255)
        mapping[4] = (255, 255, 255)
        mapping[5] = (255, 0, 255)
        mapping[6] = (0, 255, 255)
    elif num_classes == 8:
        mapping[1] = (0, 255, 0)
        mapping[2] = (255, 0, 0)
        mapping[3] = (0, 0, 255)
        mapping[4] = (255, 255, 255)
        mapping[5] = (255, 0, 255)
        mapping[6] = (0, 255, 255)
        mapping[7] = (255, 255, 0)
    else:
        raise NotImplementedError

    convert_rgb_to_onehot_parallel(mask_dir, mapping)