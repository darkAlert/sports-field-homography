import os
import pickle
import json
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.multiprocessing import Process, Queue, Event, set_start_method
from queue import Empty

from models import Reconstructor
from utils.postprocess import preds_to_masks, onehot_to_image, overlay, draw_text
from utils.config import parse_config, get_prediction_args, replace_args
from utils.logger import get_logger
from utils.dataset import BasicDataset, VideoDataset, split_on_train_val, open_court_template, open_court_poi


def save_mask_as_png(mask, dst_dir, name, postfix='mask'):
    dst_subdir = os.path.join(dst_dir, postfix)
    if not os.path.exists(dst_subdir):
        os.makedirs(dst_subdir)
    dst_path = os.path.join(dst_subdir, name + '.png')
    cv2.imwrite(dst_path, mask)

def save_mask_as_pickle(mask, writer, dst_dir, name, postfix='mask'):
    if writer is None:
        dst_subdir = os.path.join(dst_dir, postfix)
        if not os.path.exists(dst_subdir):
            os.makedirs(dst_subdir)
        dst_path = os.path.join(dst_subdir, 'data.pkl')
        writer = open(dst_path, 'wb+')

    _, buf = cv2.imencode('.png', mask)
    pickle.dump([name, buf], writer)

    return writer

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Events():
    def __init__(self):
        self.termination = Event()
        self.predict_done = Event()
        self.transfer_done = Event()

class Queues():
    def __init__(self):
        self.preds_gpu = Queue(5)
        self.preds_cpu = Queue(30)

class Workers():
    @staticmethod
    def predict(net, data_loader, device, queues, events, consistency=False, poi=False):
        '''
        Make predictions for given data
        '''
        net.eval()

        with torch.no_grad():
            for batch in data_loader:
                imgs = batch['image'].to(device=device, dtype=torch.float32)
                preds = net.predict(imgs, consistency=consistency, project_poi=poi)
                preds['name'] = batch['name']
                if 'orig_img' in batch:
                    preds['orig_img'] = batch['orig_img']
                queues.preds_gpu.put(preds)

                if events.termination.is_set():
                    break

        events.predict_done.set()
        events.termination.wait()

    @staticmethod
    def transfer_gpu_to_cpu(queues, events, req_outputs, mask_classes=4):
        '''
        Transfer the data from GPU to CPU
        '''
        while True:
            try:
                preds = queues.preds_gpu.get(timeout=0.1)
            except Empty:
                if events.predict_done.is_set():
                    break
                else:
                    continue

            # Post-processing:
            if 'segm_mask' in req_outputs:
                preds['segm_mask'] = preds_to_masks(preds['logits'], mask_classes)
            if 'logits' in preds:
                del preds['logits']
            if 'warp_mask' in req_outputs:
                preds['warp_mask'] = preds['warp_mask'].cpu().numpy().astype(np.uint8)
            else:
                if 'warp_mask' in preds:
                    del preds['warp_mask']
            if 'theta' in req_outputs:
                preds['theta'] = preds['theta'].cpu().numpy()
            else:
                if 'theta' in preds:
                    del preds['theta']
            if 'consist_score' in preds:
                preds['consist_score'] = preds['consist_score'].cpu().numpy()
            else:
                if 'consist_score' in preds:
                    del preds['consist_score']
            if 'poi' in req_outputs:
                preds['poi'] = preds['poi'].cpu().numpy()
            else:
                if 'poi' in preds:
                    del preds['poi']

            queues.preds_cpu.put(preds)

        events.transfer_done.set()
        events.termination.wait()


def process(num_data_workers=4):
    '''
    Make predictions for given data and perform post-processing
    '''
    try:
        set_start_method('spawn')
    except RuntimeError:
        assert False

    # Get params from command line:
    args = get_prediction_args()

    # Read config (yaml-file):
    if args.conf_path is None:
        args.conf_path = os.path.join(os.path.dirname(args.load), 'conf.yaml')
    if not os.path.isfile(args.conf_path):
        args.conf_path = None

    # Replace params with ones from yaml config (if it is given):
    if args.conf_path is not None:
        print ('Reading params from {}...'.format(args.conf_path))
        conf = parse_config(args.conf_path)
        ignore_keys = ['conf_path', 'batchsize', 'court_img', 'court_poi', 'img_dir', 'court_size', 'warp_size', 'load']
        args = replace_args(args, conf, ignore_keys=ignore_keys)

    # Set resolution not lower than args.out_size:
    args.out_size = tuple(args.out_size)
    if args.court_size[0] < args.out_size[0]:
        args.court_size = args.out_size
    if args.warp_size[0] < args.out_size[0]:
        args.warp_size = args.out_size

    # Prepare output:
    req_outputs = {n: True for n in args.req_outputs.split(',')}
    if not os.path.exists(args.dst_dir):
        os.makedirs(args.dst_dir)
    json_writer, pickle_writer = None, None

    if args.video_path is not None and len(args.video_path) > 0:
        game_name = os.path.basename(os.path.dirname(args.video_path))
    else:
        game_name  = os.path.basename(args.img_dir)

    # Check flags:
    project_poi = True if 'poi' in req_outputs else False
    consistency = True if 'consistency' in req_outputs else False
    keep_orig_img = True if 'debug' in req_outputs else False
    if 'debug' in req_outputs and 'warp_mask' not in req_outputs:
        req_outputs['warp_mask'] = True
    args.use_warper = True if 'warp_mask' in req_outputs or consistency else False

    assert (consistency and args.use_unet) or ~consistency
    assert (project_poi and args.use_warper) or ~project_poi

    # Log:
    logger = get_logger(format='%(message)s', write_date=False)

    # CUDA or CPU:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the court template image and the court points of interest:
    court_img = open_court_template(args.court_img,
                                    num_classes=args.mask_classes,
                                    size=args.court_size,
                                    batch_size=args.batchsize)
    court_img = court_img.to(device=device)
    court_poi = open_court_poi(args.court_poi, args.batchsize)
    court_poi = court_poi.to(device=device)

    # Init the Reconstructor network:
    net = Reconstructor(court_img, court_poi,
                        target_size = args.target_size,
                        mask_classes = args.mask_classes,
                        use_unet = args.use_unet,
                        unet_bilinear = args.unet_bilinear,
                        unet_size = args.unet_size,
                        use_resnet = args.use_resnet,
                        resnet_name = args.resnet_name,
                        resnet_input = args.resnet_input,
                        use_warper = args.use_warper,
                        warp_size = args.warp_size,
                        warp_with_nearest = True)
    net.to(device=device)
    net.load_state_dict(torch.load(args.load, map_location=device))

    # Prepare dataset:
    assert args.img_dir is not None or args.video_path is not None, \
           'img_dir and video_path cannot be both None'
    if args.img_dir is not None:
        # Use images as source:
        # ids, _ = split_on_train_val(args.img_dir, val_names=[])
        ids = [name for name in os.listdir(args.img_dir) if os.path.isfile(os.path.join(args.img_dir, name))]
        data = BasicDataset(ids, args.img_dir, None, None, None,
                            args.mask_classes, use_uv=False, target_size=args.target_size, keep_orig_img=keep_orig_img)
        data_loader = DataLoader(data, batch_size=args.batchsize, shuffle=False,
                                 num_workers=num_data_workers, pin_memory=True, drop_last=False)
        n_data = len(data)
    else:
        # Use video as source:
        data = VideoDataset(path=args.video_path, target_size=args.target_size, keep_orig_img=keep_orig_img)
        data_loader = DataLoader(data, batch_size=args.batchsize, shuffle=False,
                                 num_workers=1, pin_memory=True, drop_last=False)
        n_data = len(data)

    logger.info(f'''Start making predictions:
            Model file:        {args.load}
            Device:            {device.type}
            Images dir:        {args.img_dir}
            Video path:        {args.video_path}
            Num images:        {n_data}
            Batch size:        {args.batchsize}
            Dest dir:          {args.dst_dir}
            Required outputs:  {req_outputs}
            Mask type:         {args.mask_type}
            Mask save format:  {args.mask_save_format}
            Consistency:       {consistency}
            Use warper:        {args.use_warper}
            ResNet input size: {args.target_size}
            UNET input size:   {args.unet_size}
            Court img size:    {args.court_size}
            Warping size:      {args.warp_size}
            Output size:       {args.out_size}
        ''')

    # Run workers:
    queues = Queues()
    events = Events()
    workers = [Process(target=Workers.predict, args=(net, data_loader, device, queues, events, consistency, project_poi)),
               Process(target=Workers.transfer_gpu_to_cpu, args=(queues, events, req_outputs, net.mask_classes))]
    for w in workers:
        w.start()

    # Make and save the predictions:
    with tqdm(total=n_data, desc=f'Processing', unit='img') as pbar:
        while True:
            try:
                # Try to get the predictions:
                preds = queues.preds_cpu.get(timeout=0.1)
            except Empty:
                # Check if all work is done:
                if events.transfer_done.is_set():
                    events.termination.set()
                    break
                else:
                    continue
            pbar.update(len(preds['name']))

            # Apply post-processing to the results:
            segm_mask, warp_mask, theta, consist_score, poi = None, None, None, None, None

            # Get raw predictions:
            if 'segm_mask' in preds:
                segm_mask = preds['segm_mask']
            if 'warp_mask' in preds and 'warp_mask' in req_outputs:
                warp_mask = preds['warp_mask']
            if 'theta' in preds:
                theta = preds['theta']
            if 'consist_score' in preds:
                consist_score = preds['consist_score']
            if 'poi' in preds:
                poi = preds['poi']

            # Convert masks to required format:
            if args.mask_type == 'rgb':
                if segm_mask is not None:
                    segm_mask = onehot_to_image(segm_mask, net.mask_classes)
                if warp_mask is not None:
                    warp_mask = onehot_to_image(warp_mask, net.mask_classes)
            elif args.mask_type == 'bin':
                if segm_mask is not None:
                    segm_mask = (segm_mask > 0) * 255
                if warp_mask is not None:
                    warp_mask = (warp_mask > 0) * 255
            elif args.mask_type == 'gray':
                pass    # it is already in gray
            else:
                raise NotImplementedError

            # Resize masks to required resolution:
            if segm_mask is not None:
                if segm_mask.shape[0] != args.out_size[0] or segm_mask.size[1] != args.out_size[1]:
                    masks = []
                    for m in segm_mask:
                        masks.append(cv2.resize(m, args.out_size, interpolation=cv2.INTER_NEAREST))
                    segm_mask = np.stack(masks, axis=0)
            if warp_mask is not None:
                if warp_mask.shape[0] != args.out_size[0] or warp_mask.size[1] != args.out_size[1]:
                    masks = []
                    for m in warp_mask:
                        masks.append(cv2.resize(m, args.out_size, interpolation=cv2.INTER_NEAREST))
                    warp_mask = np.stack(masks, axis=0)

            # Save outputs:
            names = preds['name']
            for i, n in enumerate(names):
                t = n.split('/')
                if len(t) == 2:
                    subdir, name = t[0], t[1]
                else:
                    name = t[0]
                    subdir = ''

                if segm_mask is not None:
                    if args.mask_save_format == 'png':
                        save_mask_as_png(segm_mask[i], args.dst_dir, name, postfix='court/segm_mask')
                    elif args.mask_save_format == 'pickle':
                        save_mask_as_pickle(segm_mask[i], pickle_writer, args.dst_dir, name, postfix='court/segm_mask')
                    else:
                        raise NotImplementedError

                if warp_mask is not None:
                    if args.mask_save_format == 'png':
                        save_mask_as_png(warp_mask[i], args.dst_dir, name, postfix='court/warp_mask')
                    elif args.mask_save_format == 'pickle':
                        save_mask_as_pickle(warp_mask[i], pickle_writer, args.dst_dir, name, postfix='court/warp_mask')
                    else:
                        raise NotImplementedError

                if theta is not None or consist_score is not None or poi is not None:
                    if json_writer is None:
                        dst_path = os.path.join(args.dst_dir, '{}_court_processing.json'.format(game_name))
                        json_writer = open(dst_path,'w+')

                    # Write to json:
                    outputs = {}
                    if consist_score is not None:
                        outputs['score'] = float('{:5f}'.format(consist_score[i].item()))
                    if theta is not None:
                        outputs['theta'] = theta[i]
                    if poi is not None:
                        outputs['poi'] = poi[i]
                    json.dump({name: outputs}, json_writer, cls=NumpyEncoder)
                    json_writer.write('\n')

                if 'debug' in req_outputs:
                    orig_img = preds['orig_img'][i].numpy()

                    if warp_mask is not None:
                        mask = warp_mask[i]
                    elif segm_mask is not None:
                        mask = segm_mask[i]
                    else:
                        mask = None

                    if mask is not None:
                        if mask.shape[0:2] != orig_img.shape[0:2]:
                            mask = cv2.resize(mask, (orig_img.shape[1], orig_img.shape[0]), cv2.INTER_NEAREST)
                        if args.mask_type != 'rgb':
                            mask = onehot_to_image(mask, args.mask_classes)[0]
                        debug_img = overlay(orig_img, mask)
                    else:
                        debug_img = orig_img

                    if poi is not None:
                        img_H, img_W = orig_img.shape[0:2]
                        for pi, pts in enumerate(poi[i]):
                            if pts[0] < 0 or pts[0] >= img_W or pts[1] < 0 or pts[0] >= img_H:
                                continue
                            x, y = int(round(pts[0] * img_W)), int(round(pts[1] * img_H))
                            debug_img = cv2.circle(debug_img, (x, y), 3, color=(255,255,255), thickness=2)
                            draw_text(debug_img, text=str(pi), pos=(x+3, y+3), color=(128, 128, 255), scale=1)

                    if consist_score is not None:
                        draw_text(debug_img, text='{:4f}'.format(consist_score[i]), pos=(15, 15), color=(0, 255, 0), scale=0.75)

                    dst_subdir = os.path.join(args.dst_dir, 'court/debug')
                    if not os.path.exists(dst_subdir):
                        os.makedirs(dst_subdir)
                    dst_path = os.path.join(dst_subdir, name + '.jpeg')
                    cv2.imwrite(dst_path, debug_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

            # Free memory:
            del preds

    # Reformat output json files:
    json_writer.close()
    path = os.path.join(args.dst_dir, '{}_court_processing.json'.format(game_name))
    output = {k: v for line in open(path, 'r') for k, v in json.loads(line).items()}
    output['model'] = os.path.basename(os.path.dirname(args.load))
    new_path = os.path.join(args.dst_dir, '{}_court.json'.format(game_name))
    with open(new_path, 'w') as file:
        json.dump(output, file, cls=NumpyEncoder, indent=2)
    os.remove(path)

    # Free memory:
    del net, data_loader, data, logger
    torch.cuda.empty_cache()

    print ('Processing completed!')


if __name__ == "__main__":
    try:
        process()
    except:
        torch.cuda.empty_cache()
        raise
