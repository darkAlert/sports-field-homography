import cv2
import numpy as np
import torch
import torch.nn.functional as F


def preds_to_masks(preds, n_classes=1, to_ndaray=True):
    # Predictions to labels:
    if n_classes > 1:
        probs = F.softmax(preds, dim=1)
        masks = torch.argmax(probs, dim=1)
    else:
        masks = torch.sigmoid(preds)

    if to_ndaray:
        masks = masks.type(torch.IntTensor).cpu().numpy().astype(np.uint8)

    return masks


def onehot_to_image(masks, n_classes=4):
    '''
    Convert grayscale mask to RGB image
    '''
    if masks.ndim == 2:
        masks = np.expand_dims(masks, 0)    # add batch dim
    masks = np.expand_dims(masks, -1)       # add last dim, need for np.all()

    # Generate mapping:
    mapping = {}
    if n_classes == 4:
        mapping[1] = (0, 255, 0)
        mapping[2] = (255, 0, 0)
        mapping[3] = (0, 0, 255)
    elif n_classes == 7:
        mapping[1] = (0, 255, 0)
        mapping[2] = (255, 0, 0)
        mapping[3] = (0, 0, 255)
        mapping[4] = (255, 255, 255)
        mapping[5] = (255, 0, 255)
        mapping[6] = (0, 255, 255)
    elif n_classes == 8:
        mapping[1] = (0, 255, 0)
        mapping[2] = (255, 0, 0)
        mapping[3] = (0, 0, 255)
        mapping[4] = (255, 255, 255)
        mapping[5] = (255, 0, 255)
        mapping[6] = (0, 255, 255)
        mapping[7] = (255, 255, 0)
    else:
        raise NotImplementedError

    # Apply mapping:
    rgb_masks = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2], 3), dtype=np.uint8)
    for id, color in mapping.items():
        rgb_masks[np.all(masks == id, axis=3)] = color

    return rgb_masks

def overlay(frame, mask):
    m = cv2.inRange(mask, (0, 0, 0), (0, 0, 0))
    m = cv2.merge([m, m, m])
    overlaid = (frame & m) + mask*0.5 + (frame & (255 - m))*0.5

    return np.ascontiguousarray(overlaid.astype('uint8'))

def draw_text(img, text, pos, color=(255,255,255), scale=0.75, lineType=1, font=cv2.FONT_HERSHEY_COMPLEX_SMALL):
    '''
    :pos: (x,y) position of the text in the image
    '''
    cv2.putText(img, text, pos, font, scale, color, lineType)

# def transform():
#     from torchvision import transforms
#     # Transform:
#     tf = transforms.Compose(
#         [
#             transforms.ToPILImage(),
#             transforms.Resize((target_size), interpolation=Image.NEAREST),
#             transforms.ToTensor()
#         ]
#     )
#     for p in probs:
#         p = tf(p)