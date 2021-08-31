import cv2
import numpy as np
import torch
import kornia


class Warper:
    def __init__(self, size, cuda=True):
        self.warper = kornia.HomographyWarper(size[1], size[0], mode='nearest', normalized_coordinates=True)
        self.device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')

    def warp(self, theta, proj):
        proj = torch.from_numpy(proj).type(torch.DoubleTensor).permute(2, 0, 1).unsqueeze(0)
        theta = torch.from_numpy(theta).unsqueeze(0)
        proj = proj.to(self.device)
        theta = theta.to(self.device)
        proj = self.warper(proj, theta)[0]
        proj = proj.permute(1, 2, 0).cpu().numpy()

        return proj


def transform_poi(theta, poi, normalize=False):
    if poi.ndim == 3:
        proj_poi = cv2.perspectiveTransform(poi, theta)[0]
    else:
        proj_poi = cv2.perspectiveTransform(np.expand_dims(poi, axis=0), theta)[0]
    if normalize:
        proj_poi = proj_poi / 2.0 + 0.5
    return proj_poi


def map_frame_to_court(theta_f2c, frame_loc, frame_size=None):
    if not isinstance(frame_loc, np.ndarray):
        frame_loc = np.array([frame_loc], dtype=np.float32)

    if frame_size is not None:
        frame_loc[:,0] = (frame_loc[:,0] / frame_size[0] - 0.5)  * 2.0
        frame_loc[:,1] = (frame_loc[:,1] / frame_size[1] - 0.5)  * 2.0

    return transform_poi(theta_f2c, frame_loc, normalize=True)


def map_court_to_frame(theta_c2f, court_loc, court_size=None):
    if not isinstance(court_loc, np.ndarray):
        court_loc = np.array([court_loc], dtype=np.float32)

    if court_size is not None:
        court_loc[:,0] = (court_loc[:,0] / court_size[0] - 0.5) * 2.0
        court_loc[:,1] = (court_loc[:,1] / court_size[1] - 0.5) * 2.0

    return transform_poi(theta_c2f, court_loc, normalize=True)