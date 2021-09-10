import os
import json
import pickle
import argparse
from tqdm import tqdm
import cv2
import numpy as np

from subprocess import PIPE, run
import shutil

import torch
import kornia

from utils.postprocess import onehot_to_image, overlay, draw_text
from utils.dataset import open_court_template, open_court_poi


class PredictionReader:
    def __init__(self, path):
        self.preds = None
        with open(path, 'r') as file:
            self.preds = json.load(file)

    def get(self):
        for name, p in self.preds.items():
            yield name, p

class VideoReader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)

    def __len__(self):
        if self.cap is not None:
            return int(cv2.VideoCapture.get(self.cap, cv2.CAP_PROP_FRAME_COUNT))
        else:
            return 0

    def get(self):
        assert self.cap.isOpened() == True
        f_num = 0

        while True:
            ret, frame = self.cap.read()
            if ret == False:
                break
            yield f_num, frame
            f_num += 1

        self.cap.release()

class MaskReader:
    def __init__(self, path=None, from_preds=None):
        assert path is not None or from_preds is not None
        self.entries = []
        if path is not None:
            with open(path, 'rb') as file:
                while True:
                    try:
                        self.entries.append(pickle.load(file))
                    except EOFError:
                        break
        else:
            for k,v in from_preds.get():
                self.entries.append([k,None])

    def get(self, decode=False):
        for name, buf in self.entries:
            if decode:
                buf = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            yield name, buf

    @staticmethod
    def decode(buf):
        return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


def vizualize(video_path, preds_path, dst_dir, field_path, masks_path=None,
              mask_classes=4, out_size=(1280, 720), fps=30, score_threshold=0.1, overlay_threshold=None):
    chunk_size = 10000
    out_W, out_H = out_size[:]

    # Open sources:
    preds = PredictionReader(preds_path)
    video = VideoReader(video_path)
    if masks_path is not None:
        masks = MaskReader(masks_path)
    else:
        masks = MaskReader(from_preds=preds)

    n_frames = len(video)

    # Load the court template image:
    court_img = open_court_template(field_path, mask_classes, (out_W, out_H), 1)
    warper = kornia.HomographyWarper(out_H, out_W, mode='nearest', normalized_coordinates=True)

    temp_dir = os.path.join(dst_dir, '_temp')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    chunk_list_path = os.path.join(temp_dir, 'chunks.txt')
    chunk_list_file = open(chunk_list_path, 'w')
    chunk_i, counter = 0, 0
    dst_subdir = None

    # Process:
    with tqdm(total=n_frames, desc=f'Processing', unit='img') as pbar:
        for (f_num, frame), (p_name, pred), (m_name, segm_mask) in zip(video.get(), preds.get(), masks.get()):
            assert p_name == m_name and int(p_name) == f_num

            # Create new chunk folder:
            if counter == 0:
                dst_subdir = os.path.join(temp_dir, '_chunk{}/'.format(chunk_i))
                if not os.path.exists(dst_subdir):
                    os.makedirs(dst_subdir)

            # Get:
            score = pred['score']
            theta = torch.FloatTensor(pred['theta'])
            if score < score_threshold:
                # Mask via warping:
                mask = warper(court_img, theta)[0]
                mask = mask * mask_classes
                mask = mask.type(torch.IntTensor).numpy().astype(np.uint8)
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
                # Mask via segmentation:
                if segm_mask is not None:
                    mask = MaskReader.decode(segm_mask)
                else:
                    mask = None

            # Post-process:
            if mask is not None:
                mask = onehot_to_image(mask, mask_classes)[0]

            # Resize:
            if mask is not None and (mask.shape[0] != out_H or mask.shape[1] != out_W):
                mask = cv2.resize(mask, (out_W, out_H), interpolation=cv2.INTER_NEAREST)

            # Draw:
            if mask is not None and overlay_threshold is None or \
                    (overlay_threshold is not None and score < overlay_threshold):
                frame = overlay(frame, mask)
            draw_text(frame, text='{:4f}'.format(score), pos=(15, 15), color=color, scale=0.75)

            # Save frame as image:
            # dst_path = os.path.join(temp_dir, p_name + '.jpeg')
            # cv2.imwrite(dst_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            dst_path = os.path.join(dst_subdir, p_name + '.png')
            cv2.imwrite(dst_path, frame)

            # Convert chunk images to mp4:
            counter += 1
            if counter >= chunk_size:
                dst_path = os.path.join(temp_dir, 'chunk{}.mp4'.format(chunk_i))
                chunk_list_file.write('file ' + dst_path + '\n')
                cmd = 'ffmpeg -pattern_type glob -framerate {} -f image2 -i \'{}*.png\' {}'.format(fps, dst_subdir, dst_path)
                run(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
                shutil.rmtree(dst_subdir)
                chunk_i += 1
                counter = 0

            pbar.update(1)

    # Convert last chunk images to mp4:
    if counter != 0:
        dst_path = os.path.join(temp_dir, 'chunk{}.mp4'.format(chunk_i))
        chunk_list_file.write('file ' + dst_path + '\n')
        cmd = 'ffmpeg -pattern_type glob -framerate {} -f image2 -i \'{}*.png\' {}'.format(fps, dst_subdir, dst_path)
        run(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
        shutil.rmtree(dst_subdir)
    chunk_list_file.close()

    # Concatenate chunk videos into one output video:
    dst_video_path = os.path.join(dst_dir, 'output.mp4')
    if os.path.exists(dst_video_path):
        os.remove(dst_video_path)
    cmd = 'ffmpeg -f concat -safe 0 -i {} -c copy {}'.format(chunk_list_path, dst_video_path)
    run(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    shutil.rmtree(temp_dir)
    print ('Output video has been saved to', dst_video_path)


    print('All done!')

def get_args():
    parser = argparse.ArgumentParser(description='Reconstructor')
    parser.add_argument('--video_path', type=str, default=None)
    parser.add_argument('--preds_path', type=str, default=None)
    parser.add_argument('--dst_dir', type=str, default=None)
    parser.add_argument('--masks_path', type=str, default=None)
    parser.add_argument('--field_path', type=str, default='./assets/mask_ncaa_v4_nc4_m_onehot.png')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--score_threshold', type=float, default=0.17)
    parser.add_argument('--overlay_threshold', type=float, default=None)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    vizualize(args.video_path, args.preds_path, args.dst_dir, args.field_path, args.masks_path,
              fps=args.fps, score_threshold=args.score_threshold, overlay_threshold=args.overlay_threshold)

