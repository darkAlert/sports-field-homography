import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


class PoIHorizontalFlip(torch.nn.Module):
    """
    Horizontally flip the given Points of Interest (PoI) randomly with a given probability.

    Args:
        p (float): probability of the PoI being flipped. Default value is 0.5
    """
    mapping = None

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        if PoIHorizontalFlip.mapping is None:
            PoIHorizontalFlip.mapping = PoIHorizontalFlip.flipped_poi_mapping()

    @staticmethod
    def flipped_poi_mapping():
        mapping = torch.zeros(28, dtype=torch.int8)
        for i in range(0, 4):
            mapping[i] = i
        for i in range(0, 4):
            mapping[4 + i] = 51 - i
        for i in range(0, 14):
            mapping[8 + i] = 45 - i
        for i in range(0, 2):
            mapping[22 + i] = 47 - i
        for i in range(0, 4):
            mapping[24 + i] = 31 - i

        return mapping

    def forward(self, input):
        """
        Args:
            PoI Tensor: PoI to be flipped.

        Returns:
            PoI Tensor: Randomly flipped PoI.
        """
        poi, nonzeros = input['poi'], input['nonzeros']

        if torch.rand(1) < self.p:
            t_poi = torch.empty(poi.shape, dtype=poi.dtype)
            t_nonzeros = torch.empty(nonzeros.shape, dtype=nonzeros.dtype)
            n = PoIHorizontalFlip.mapping.shape[0]
            for idx1 in range(n):
                idx2 = PoIHorizontalFlip.mapping[idx1]
                t_poi[idx1, 0] = 1.0 - poi[idx2, 0]
                t_poi[idx1, 1] = poi[idx2, 1]
                t_poi[idx2, 0] = 1.0 - poi[idx1, 0]
                t_poi[idx2, 1] = poi[idx1, 1]
                t_nonzeros[idx1] = nonzeros[idx2]
                t_nonzeros[idx2] = nonzeros[idx1]

            return t_poi, t_nonzeros

        return poi, nonzeros

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


def make_apperance_transform(aug):
    '''
    Make apperance transformations for data augmentation
    '''
    assert aug is not None
    trans = []

    # Select transformations:
    if 'jitter' in aug:
        brightness = 0.35                # params by default
        contrast = 0.35
        saturation = 0.25
        hue = 0.25
        jitter = aug['jitter']
        if 'brightness' in jitter: brightness = jitter['brightness']
        if 'contrast' in jitter: contrast = jitter['contrast']
        if 'saturation' in jitter: saturation = jitter['saturation']
        if 'hue' in jitter: hue = jitter['hue']
        trans.append(transforms.ColorJitter(brightness=brightness,
                                            contrast=contrast,
                                            saturation=saturation,
                                            hue=hue))
    if 'blur' in aug:
        kernel_size = aug['blur']
        trans.append(transforms.GaussianBlur(kernel_size))

    assert len(trans) > 0, \
    'List of apperance transformations is empty. If you do not want '\
    'to use any apperance transformations, set aug[\'apperance\'] to None.'

    # Compose transformations:
    TF = transforms.Compose(trans)

    return TF

def make_geometric_transform(aug, target_widht=1280, target_height=720, interpol=Image.BILINEAR):
    '''
    Make geometric transformations for data augmentation
    '''
    assert aug is not None
    trans = []

    # Select transformations:
    if 'scale' in aug:
        scale = aug['scale']                               # scale by default = (0.5, 1.0)
        ratio = target_widht / float(target_height)
        trans.append(transforms.RandomResizedCrop((target_height,target_widht),
                                                  scale=scale,
                                                  ratio=(ratio,ratio),
                                                  interpolation=interpol))     # Image.NEAREST
    if 'hflip' in aug:
        hflip = aug['hflip']
        trans.append(transforms.RandomHorizontalFlip(hflip))

    assert len(trans) > 0, \
    'List of geometric transformations is empty. If you do not want '\
    'to use any geometric transformations, set aug[\'geometric\'] to None.'

    # Compose transformations:
    TF = transforms.Compose(trans)

    return TF

def make_points_transform(aug):
    '''
    Make points transformations for data augmentation
    '''
    assert aug is not None
    trans = []

    # Select transformations:
    if 'scale' in aug:
        raise NotImplementedError

    if 'hflip' in aug:
        hflip = aug['hflip']
        trans.append(PoIHorizontalFlip(hflip))

    assert len(trans) > 0, \
    'List of points transformations is empty. If you do not want '\
    'to use any points transformations, set aug[\'points\'] to None.'

    # Compose transformations:
    TF = transforms.Compose(trans)

    return TF

def apply_transforms(img, mask, poi=None, nonzeros=None,
                     TF_apperance=None,
                     TF_img_geometric=None,
                     TF_msk_geometric=None,
                     TF_poi_geometric=None,
                     geometric_same=True,
                     seed=42):
    '''
    :geometric_same: If True, then applies the same geometric transform to input mask
    '''
    assert TF_apperance is not None or \
          (TF_img_geometric is not None and TF_msk_geometric is not None)
    assert (TF_img_geometric is None and TF_msk_geometric is None) or \
           (TF_img_geometric is not None and TF_msk_geometric is not None)
    assert TF_poi_geometric is not None and poi is not None and nonzeros is not None or \
           TF_poi_geometric is None

    # Convert temporally to specific dtype:
    img_dtype, mask_dtype = img.dtype, mask.dtype
    img = img.to(dtype=torch.float32)
    mask = mask.to(dtype=torch.uint8)

    # Transform img:
    if TF_apperance is not None:
        img = TF_apperance(img)
    if geometric_same:
        torch.manual_seed(seed)
    if TF_img_geometric is not None:
        img = TF_img_geometric(img)

    # Transform mask with the same seed:
    if geometric_same:
        torch.manual_seed(seed)
    if TF_msk_geometric is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        mask = TF_msk_geometric(mask)

    # Transform PoI with the same seed:
    if geometric_same:
        torch.manual_seed(seed)
    if TF_poi_geometric is not None:
        input = {'poi': poi, 'nonzeros': nonzeros}
        poi, nonzeros = TF_poi_geometric(input)

    # Convert back to the original dtype:
    img = img.to(dtype=img_dtype)
    mask = mask.to(dtype=mask_dtype)

    return img, mask, poi, nonzeros


if __name__ == '__main__':
    '''
    Augmentation test
    '''
    import cv2
    from torch.utils.data import DataLoader
    from utils.dataset import BasicDataset, split_on_train_val, worker_init_fn
    from utils.postprocess import onehot_to_image, overlay

    # Paths:
    img_dir = '/media/darkalert/c02b53af-522d-40c5-b824-80dfb9a11dbb/boost/datasets/court_segmentation/test/test_aug/frames/'
    mask_dir = '/media/darkalert/c02b53af-522d-40c5-b824-80dfb9a11dbb/boost/datasets/court_segmentation/test/test_aug/masks/'
    anno_dir = '/media/darkalert/c02b53af-522d-40c5-b824-80dfb9a11dbb/boost/datasets/court_segmentation/test/test_aug/anno/'
    anno_keys = ['poi']
    n_classes = 4
    size = (640, 320)
    batchsize = 2
    dst_dir = '/media/darkalert/c02b53af-522d-40c5-b824-80dfb9a11dbb/boost/datasets/court_segmentation/test/test_aug/_results/'
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # Augmentations:
    jitter = {'brightness': 0.35, 'contrast': 0.35, 'saturation': 0.25, 'hue': 0.25}
    apperance = {'jitter': jitter, 'blur': 5}
    geometric = {'hflip': 0.5}
    # aug = {'apperance': apperance, 'geometric': geometric}
    aug = {'geometric': geometric}

    # Prepare dataset:
    ids, _ = split_on_train_val(img_dir, val_names=[])
    # ids = [ids[0]]
    data = BasicDataset(ids, img_dir, mask_dir, anno_dir, anno_keys, n_classes, size, aug=aug)
    loader = DataLoader(data, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True,
                        drop_last=False, worker_init_fn=worker_init_fn)

    # Aplly augmentations:
    for iter in range(10):
        for bi, batch in enumerate(loader):
            img, mask = batch['image'], batch['mask']

            # Postprocess:
            img = img.cpu().numpy().astype('float32')
            img = np.transpose(img, (0, 2, 3, 1))
            img = (img * 255.0).astype('uint8')

            mask = mask.cpu().numpy().astype('float32')
            mask = onehot_to_image(mask, n_classes)

            if 'poi' in batch:
                poi = batch['poi']
                nonzeros = batch['nonzeros']
                num_nonzero = batch['num_nonzero']

                poi = poi.cpu().numpy().astype('float32')


            for i, (out_img, out_mask) in enumerate(zip(img, mask)):
                out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)

                # Draw PoI:
                if poi is not None:
                    h, w = out_img.shape[0:2]
                    out_poi = poi[i]
                    for pi, pts in enumerate(out_poi):
                        x, y = int(round(pts[0]*w)), int(round(pts[1]*h))
                        out_img = cv2.circle(out_img, (x, y), 3, (0, 255, 255), 2)
                        cv2.putText(out_img, str(pi), (x,y),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (64, 255, 0), 1)
                    out_img = cv2.resize(out_img, (0,0), fx=2, fy=2)

                # overlaid = overlay(out_img, out_mask)
                overlaid = out_img
                out_path = os.path.join(dst_dir, '{}_{}_{}.png'.format(iter, bi, i))
                cv2.imwrite(out_path, overlaid)



    print ('Done!')

    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),








