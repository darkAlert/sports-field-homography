import os
from os import listdir
import json
import numpy as np
import cv2
from glob import glob

import torch
from torch.utils.data import Dataset
from PIL import Image

from utils.augmentation import make_apperance_transform, make_geometric_transform, make_points_transform, apply_transforms


def worker_init_fn(worker_id):
    '''
    Use this function to set the numpy seed of each worker
    For example, loader = DataLoader(..., worker_init_fn=worker_init_fn)
    '''
    np.random.seed()
    # seed = np.random.get_state()[1][0] + worker_id

def split_on_train_val(img_dir, val_names, only_ncaam=False):
    '''
    Split a dataset on training and validation ids
    '''
    names = [n for n in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, n))]
    train_ids = []
    val_ids = []

    for name in names:
        subdir = os.path.join(img_dir, name)
        ids = [os.path.join(name,file) for file in listdir(subdir) if not file.endswith('.')]
        if any(name == n for n in val_names):
            val_ids += ids
        else:
            if only_ncaam and name[0] == '2':
                print('Skip', name)
                continue
            train_ids += ids

    return train_ids, val_ids

def open_court_template(path, num_classes, size=None, batch_size=1):
    '''
    Load the court template image that will be projected by affine matrix from STN
    '''
    template = Image.open(path)
    if size is not None:
        template = template.resize(size, resample=Image.NEAREST)
    template = np.array(template) / float(num_classes)
    template_tensor = torch.from_numpy(template).type(torch.FloatTensor)

    while template_tensor.ndim < 4:
        template_tensor = template_tensor.unsqueeze(0)
    template_tensor = template_tensor.repeat(batch_size, 1, 1, 1)

    return template_tensor

def open_court_poi(path, batch_size=1, normalize=True, homogeneous=False):
    '''
    Load the points of interest of court template
    '''
    points = None

    with open(path) as f:
        try:
            points_data = json.load(f)
            points_raw = points_data['points']
            ranges = points_data['ranges']
            assert ranges[0] == 1.0 and ranges[1] == 1.0
            points = []

            for p in points_raw:
                if normalize:
                    x, y = (p['coords'][0] - 0.5) * 2, (p['coords'][1] - 0.5) * 2
                else:
                    x, y = p['coords'][0], p['coords'][1]

                if homogeneous:
                    points.append((x, y, 1.0))
                else:
                    points.append((x, y))
            points = np.array(points)

        except Exception as e:
            raise ValueError(f'Cannot read {path}: {str(e)}')

    points_tensor = torch.from_numpy(points).type(torch.FloatTensor)
    points_tensor = points_tensor.unsqueeze(0)
    points_tensor = points_tensor.repeat(batch_size, 1, 1)

    return points_tensor


class BasicDataset(Dataset):
    def __init__(self, ids, img_dir, mask_dir=None, anno_dir=None, anno_keys=None,
                 num_classes=1, target_size=(1280,720), aug=None, keep_orig_img=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.anno_dir = anno_dir
        self.ids = ids
        self.target_size = target_size
        self.num_classes = num_classes
        self.aug = aug
        self.anno_keys = anno_keys
        self.keep_orig_img = keep_orig_img
        self.TF_apperance = None
        self.TF_img_geometric = None
        self.TF_msk_geometric = None
        self.TF_poi_geometric = None

        assert anno_dir != None and anno_keys != None or anno_dir == None

        # Get transforms:
        if self.aug is not None:
            if 'apperance' in self.aug and self.aug['apperance'] is not None:
                self.TF_apperance = make_apperance_transform(self.aug['apperance'])
            if 'geometric' in self.aug and self.aug['geometric'] is not None:
                self.TF_img_geometric = make_geometric_transform(self.aug['geometric'],
                                                                 target_size[0],
                                                                 target_size[1],
                                                                 Image.BILINEAR)
                self.TF_msk_geometric = make_geometric_transform(self.aug['geometric'],
                                                                 target_size[0],
                                                                 target_size[1],
                                                                 Image.NEAREST)
                if anno_keys is not None and 'poi' in anno_keys:
                    self.TF_poi_geometric = make_points_transform(self.aug['geometric'])

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess_img(pil_img, target_size, normalize=True):
        pil_img = pil_img.resize(target_size)
        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_nd = img_nd.transpose((2, 0, 1))
        if normalize:
            img_nd = img_nd / 255

        # To tensor:
        img_tensor = torch.from_numpy(img_nd).type(torch.FloatTensor)

        return img_tensor

    @staticmethod
    def preprocess_mask(pil_mask, target_size):
        pil_mask = pil_mask.resize(target_size, resample=Image.NEAREST)
        mask_nd = np.array(pil_mask)
        mask_tensor = torch.from_numpy(mask_nd).type(torch.LongTensor)

        return mask_tensor

    @staticmethod
    def preprocess_poi(np_poi):
        anno_tensor = torch.from_numpy(np_poi).type(torch.FloatTensor)
        nonzeros = anno_tensor[:, 2]
        poi = anno_tensor[:, :2]
        num_nonzero = torch.count_nonzero(nonzeros, dim=0)

        return poi, nonzeros, num_nonzero

    @staticmethod
    def preprocess_weight(reproj_mse):
        '''
        Apply Sigmoid to Reprojection MSE score
        '''
        x = reproj_mse / 0.01
        x = x * 12 - 6
        x = x * 1.25 + 1
        y = 1 - 1 / (1 + np.exp(-x))
        y = np.array([y], dtype='float32')

        weight = torch.from_numpy(y).type(torch.FloatTensor)

        return weight

    @staticmethod
    def preprocess_anno(np_anno):
        anno_tensor = torch.from_numpy(np_anno).type(torch.FloatTensor)

        return anno_tensor

    def __getitem__(self, i):
        name = self.ids[i]
        name_wo_ext = name[:name.rfind('.')]
        sample = {'name': name_wo_ext}

        # Get image and mask paths:
        img_file = glob(os.path.join(self.img_dir, name))
        mask_file = glob(os.path.join(self.mask_dir, name_wo_ext + '.png')) if self.mask_dir is not None else None
        anno_file = glob(os.path.join(self.anno_dir, name_wo_ext + '.json')) if self.anno_dir is not None else None
        assert len(img_file) == 1, \
            'Either no image or multiple images found for the ID {}: {}'.format(name, img_file)
        assert mask_file is None or len(mask_file) == 1, \
            'Either no mask or multiple masks found for the ID {}: {}'.format(name_wo_ext + '.png', mask_file)
        assert anno_file is None or len(anno_file) == 1, \
            'Either no json or multiple json found for the ID {}: {}'.format(name_wo_ext + '.json', anno_file)

        # Open image and mask:
        orig_img = Image.open(img_file[0])
        assert orig_img is not None
        mask = Image.open(mask_file[0]) if mask_file is not None else None

        # Preprocess image and mask:
        img = self.preprocess_img(orig_img, self.target_size)
        mask = self.preprocess_mask(mask, self.target_size) if mask is not None else None

        # Open and preprocess annotations:
        poi, nonzeros, num_nonzero = None, None, None
        if anno_file is not None:
            with open(anno_file[0], 'r') as json_file:
                json_data = json.load(json_file)
                for k in self.anno_keys:
                    anno = np.asarray(json_data[k], dtype='float')
                    if k == 'poi':
                        poi, nonzeros, num_nonzero = self.preprocess_poi(anno)
                    elif k == 'reproj_mse':
                        sample['weight'] = self.preprocess_weight(anno)
                    else:
                        sample[k] = self.preprocess_anno(anno)

        # Augmentation:
        if self.aug is not None:
            img, mask, poi, nonzeros = apply_transforms(img, mask, poi, nonzeros,
                                                        self.TF_apperance,
                                                        self.TF_img_geometric,
                                                        self.TF_msk_geometric,
                                                        self.TF_poi_geometric,
                                                        seed=np.random.randint(2147483647))

        if mask is not None and mask.ndim == 3:
            mask = mask.squeeze(0)        # [1,h,w] -> [h,w]

        sample['image'] = img
        if mask is not None:
            sample['mask'] = mask
        if poi is not None:
            sample['poi'] = poi
        if poi is not None:
            sample['nonzeros'] = nonzeros
        if poi is not None:
            sample['num_nonzero'] = num_nonzero
        if self.keep_orig_img:
            sample['orig_img'] = cv2.cvtColor(np.array(orig_img),cv2.COLOR_RGB2BGR)

        return sample


class VideoDataset(Dataset):
    def __init__(self, path, target_size=(640,360), max_frames=None, keep_orig_img=False):
        self.cap = None
        self.path = path
        self.target_size = target_size
        self.keep_orig_img = keep_orig_img
        t = os.path.basename(path)
        self.name = t[:t.rfind('.')]
        num_frames = int(cv2.VideoCapture.get(cv2.VideoCapture(self.path), cv2.CAP_PROP_FRAME_COUNT))
        self.num_frames = min(num_frames, max_frames) if max_frames is not None else num_frames

    def __len__(self):
        return self.num_frames

    @staticmethod
    def preprocess_img(frame, target_size, normalize=True):
        # Resize:
        target_w, target_h = target_size[0], target_size[1]
        h, w = frame.shape[0:2]
        if w != target_w or h != target_h:
            inter = cv2.INTER_AREA if w > target_w else cv2.INTER_LINEAR
            frame = cv2.resize(frame, (target_w,target_h), interpolation=inter)

        if len(frame.shape) == 2:
            frame = np.expand_dims(frame, axis=2)

        # HWC to CHW
        frame = frame.transpose((2, 0, 1))
        if normalize:
            frame = frame / 255

        # To tensor:
        frame_tensor = torch.from_numpy(frame).type(torch.FloatTensor)

        return frame_tensor

    def __getitem__(self, i):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.path)

        ret, frame = self.cap.read()

        # If the frame could not be read:
        if ret == False:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.cap.read()
            if ret == False:
                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame = np.zeros((h,w,3),dtype=np.uint8)

        img = self.preprocess_img(frame, self.target_size)

        name = self.name + '/' + str(i).zfill(6)
        sample = {'image': img, 'name': name}
        if self.keep_orig_img:
            sample['orig_img'] = frame

        return sample

    def __del__(self):
        if self.cap is not None:
            self.cap.release()