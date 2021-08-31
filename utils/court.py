import json
import numpy as np
import cv2


class CourtSizes:
    '''
    Contains the constants of the court size dimensions.
    '''
    COURT_IN_PIXELS = (1280, 720)
    FRAME_IN_PIXELS = (1280, 720)
    COURT_IN_METERS = (32.2326, 17.145)
    METERS2FEET = 3.28084
    METERS2PIXELS = (COURT_IN_PIXELS[0] / COURT_IN_METERS[0],
                     COURT_IN_PIXELS[1] / COURT_IN_METERS[1])
    PIXELS2METERS = (COURT_IN_METERS[0] / COURT_IN_PIXELS[0],
                     COURT_IN_METERS[1] / COURT_IN_PIXELS[1])


class CourtMapping:
    '''
    Loads, parses and prepares the court mapping (homography) from json for each frame.
    '''
    class FrameMapping:
        '''
        Auxiliary class containing thetas and predicted score for each frame.
        '''
        def __init__(self, theta_f2c, theta_c2f, score):
            self.theta_f2c = theta_f2c           # frame to court homography
            self.theta_c2f = theta_c2f           # court to frame homography
            self.score = score

    def __init__(self, path):
        mapping_raw = CourtMapping._load_mapping(path)
        self.frames = {}
        if 'model' in mapping_raw:
            model_name = mapping_raw.pop('model')
            print ('Court homography was predicted by the {} model'.format(model_name))

        for frame_id, data in mapping_raw.items():
            score = float(data['score'])
            theta_f2c = np.array(data['theta'])[0]
            theta_c2f = np.linalg.inv(theta_f2c)
            self.frames[frame_id] = CourtMapping.FrameMapping(theta_f2c, theta_c2f, score)

    @staticmethod
    def _load_mapping(path):
        '''
        Load json file with court mapping (homography for every frame)
        '''
        with open(path, 'r') as file:
            mapping = json.load(file)
        return mapping


def load_court_mask(path, court_size, inter=None):
    '''
    Loads the court template image that will be warped by homography.
    '''
    court_mask = cv2.imread(path, cv2.IMREAD_COLOR)

    if court_mask.shape[0] != court_size[1] or court_mask.shape[1] != court_size[0]:
        if inter is None:
            inter = cv2.INTER_AREA if court_mask.shape[1] > court_size[0] else cv2.INTER_CUBIC
        court_mask = cv2.resize(court_mask, court_size, interpolation=inter)

    return court_mask


def load_court_poi(path, normalize=True, homogeneous=False):
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

    return points


