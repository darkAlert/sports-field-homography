import cv2
import numpy as np
from utils.court import load_court_mask, load_court_poi
from utils.court import CourtSizes as CS
from utils.transform import map_frame_to_court, map_court_to_frame


def map_frame_points_to_court():
    court_img = load_court_mask('./assets/template_ncaa_v4_s.png', court_size=CS.COURT_IN_PIXELS)

    # Theta (predicted homography):
    theta_f2c = np.array([
        [
          8.030766487121582, -0.22687992453575134, 9.891857147216797
        ],
        [
          3.553352117538452, 25.72734260559082, -0.09768841415643692
        ],
        [
          0.1463453769683838, 5.179210662841797, 16.56546974182129
        ]
    ])

    # Frame points:
    frame_points = np.array([[590, 418]], dtype=np.float32)

    # Transform:
    court_points = map_frame_to_court(theta_f2c, frame_points, frame_size=CS.FRAME_IN_PIXELS)
    court_points[:,0] *= CS.COURT_IN_PIXELS[0]
    court_points[:,1] *= CS.COURT_IN_PIXELS[1]


    # Draw points on the court image:
    for pt in court_points:
        x, y = int(round(pt[0])), int(round(pt[1]))
        court_img = cv2.circle(court_img, (x, y), 5, color=(255, 0, 255), thickness=-1)

    cv2.imshow('court_img', court_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def mape_court_poi_to_frame():
    COURT_POI_PATH = './assets/template_ncaa_v4_points.json'
    court_poi = load_court_poi(COURT_POI_PATH)

    # Homography (frame to court):
    theta_f2c = np.array([
        [
          5.78266048, -0.43701401,  8.0031395
        ],
        [
          3.63819695, 15.77359295, -0.46604609
        ],
        [
          0.14406031, 3.68673325, 13.25017166
        ]
    ])
    # Get court to frame homography:
    theta_c2f = np.linalg.inv(theta_f2c)

    # Transform:
    frame_poi = map_court_to_frame(theta_c2f, court_poi)
    frame_poi[:,0] *= CS.FRAME_IN_PIXELS[0]
    frame_poi[:,1] *= CS.FRAME_IN_PIXELS[1]

    # Draw points on the court image:
    frame = np.zeros((CS.FRAME_IN_PIXELS[1], CS.FRAME_IN_PIXELS[0], 3), np.uint8)

    for pt in frame_poi:
        x, y = int(round(pt[0])), int(round(pt[1]))
        frame = cv2.circle(frame, (x, y), 5, color=(255, 0, 255), thickness=-1)

    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    map_frame_points_to_court()
    mape_court_poi_to_frame()