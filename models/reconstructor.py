""" Full assembly of the parts to form the complete network """
from enum import Enum
import kornia
from kornia.geometry.linalg import transform_points
from unet.unet_parts import *
from models.resnet import resnet_stn


class Input(Enum):
    IMG = 1
    MASK = 2
    IMG_AND_MASK = 3

    @classmethod
    def parse(cls, input):
        if input is None:
            return None
        elif input == 'img':
            return cls.IMG
        elif input == 'mask':
            return cls.MASK
        elif input == 'img+mask':
            return cls.IMG_AND_MASK
        else:
            raise NotImplementedError

class Reconstructor(nn.Module):
    '''The Reconstructor model consists of the UNET, which learns
    to segment the court mask, and the Spatial Transformer Network
    (STN), which learns to predict the homography matrix based on
    the predicted segmentation mask.
    '''
    def __init__(self, court_img, court_poi,
                 target_size = (640, 360),
                 mask_classes = 4,
                 use_unet = True,
                 unet_bilinear = False,
                 unet_size = (640, 360),
                 use_resnet = True,
                 resnet_name = 'resnet34',
                 resnet_input = 'img+mask',
                 resnet_pretrained = None,
                 use_warper = True,
                 warp_size = (640,360),
                 warp_with_nearest = False):
        super(Reconstructor, self).__init__()
        assert use_unet is not None or use_resnet is not None

        # The court template image and court points of interest
        # which will be warped by the learnt homography matrix:
        self.court_img = court_img
        self.court_poi = court_poi
        self.target_size = target_size
        self.mask_classes = mask_classes
        self.use_unet = use_unet
        self.unet_size = unet_size
        self.use_resnet = use_resnet
        self.resnet_input = Input.parse(resnet_input)

        # UNet outputs the segmentation mask:
        if self.use_unet:
            n_channels = 3    # rgb
            self.inc = DoubleConv(n_channels, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            factor = 2 if unet_bilinear else 1
            self.down4 = Down(512, 1024 // factor)
            self.up1 = Up(1024, 512 // factor, unet_bilinear)
            self.up2 = Up(512, 256 // factor, unet_bilinear)
            self.up3 = Up(256, 128 // factor, unet_bilinear)
            self.up4 = Up(128, 64, unet_bilinear)
            self.outc = OutConv(64, mask_classes)

        # ResNetSTN outputs the 3x3 transformation matrix:
        if self.use_resnet:
            assert self.resnet_input is not None
            if self.resnet_input == Input.IMG:
                in_classes = 3
            elif self.resnet_input == Input.MASK:
                assert self.use_unet
                in_classes = mask_classes
            elif self.resnet_input == Input.IMG_AND_MASK:
                assert self.use_unet
                in_classes = mask_classes+3
            else:
                assert False
            self.resnet_reg = resnet_stn(resnet_name, resnet_pretrained, in_classes)

        # Homography warper:
        self.warper = None
        if use_warper:
            h, w = warp_size[1], warp_size[0]
            if warp_with_nearest is True:
                # It seems mode='nearest' has a bug when used during training
                self.warper = kornia.HomographyWarper(h, w, mode='nearest', normalized_coordinates=True)
            else:
                self.warper = kornia.HomographyWarper(h, w, normalized_coordinates=True)

    def warp(self, theta, court_img):
        '''
        Warp teamplate image by predicted homographies
        '''
        bs = theta.shape[0]
        template = court_img[0:bs]

        warped = self.warper(template, theta)

        return warped.squeeze(1)

    def transform_poi(self, theta, court_poi, normalize=True):
        ''' Transform PoI with the homography '''
        bs = theta.shape[0]
        theta_inv = torch.inverse(theta[:bs])
        poi = transform_points(theta_inv, court_poi[:bs])

        # Apply inverse normalization to the transformed PoI (from [-1,1] to [0,1]):
        if normalize:
            poi = poi / 2.0 + 0.5

        return poi

    def forward_unet(self, x):
        # Fit input size:
        if x.shape[3] != self.unet_size[0] or x.shape[2] != self.unet_size[1]:
            w, h = self.unet_size[:]
            x = F.interpolate(x,size=(h,w), mode='bilinear', align_corners=False)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x_top = self.down4(x4)
        x = self.up1(x_top, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        # Fit output size:
        if logits.shape[3] != self.target_size[0] or logits.shape[2] != self.target_size[1]:
            w,h = self.target_size[:]
            logits = F.interpolate(logits,size=(h,w), mode='nearest')

        return logits, x_top

    def forward(self, x):
        '''
        Forward pass
        '''
        ret = {}

        # UNet:
        if self.use_unet:
            ret['logits'], _ = self.forward_unet(x)

        # ResNetSTN:
        if self.use_resnet:
            if self.resnet_input == Input.IMG:
                y = x
            elif self.resnet_input == Input.MASK:
                y = ret['logits']
            elif self.resnet_input == Input.IMG_AND_MASK:
                y = torch.cat((ret['logits'], x), 1)
            else:
                raise NotImplementedError

            theta = self.resnet_reg(y)
            poi = self.transform_poi(theta, self.court_poi)
            ret['theta'] = theta
            ret['poi'] = poi

            if self.warper is not None:
                warp_mask = self.warp(theta, self.court_img)
                ret['warp_mask'] = warp_mask

        return ret

    def predict(self, x, consistency=True, project_poi=False):
        '''
        Predicts the transformation matrix (theta) from input image (x).
        If warp is True then it also warps the court_img using the predicted theta
        '''
        ret = {}

        # UNet:
        if self.use_unet:
            ret['logits'], _ = self.forward_unet(x)

        # ResNetSTN:
        if self.use_resnet:
            if self.resnet_input == Input.IMG:
                y = x
            elif self.resnet_input == Input.MASK:
                y = ret['logits']
            elif self.resnet_input == Input.IMG_AND_MASK:
                y = torch.cat((ret['logits'], x), 1)
            else:
                raise NotImplementedError

            theta = self.resnet_reg(y)
            ret['theta'] = theta

            # Warp the court template:
            if self.warper is not None:
                ret['warp_mask'] = self.warp(theta, self.court_img) * self.mask_classes

                # Calculate consistency score:
                if consistency and self.use_unet:
                    logits, warp_mask = ret['logits'], ret['warp_mask']

                    # Resize warp_mask:
                    if logits.shape[2:4] != warp_mask.shape[1:3]:
                        warp_mask = torch.unsqueeze(warp_mask,1)
                        h, w = logits.shape[2:4]
                        warp_mask = F.interpolate(warp_mask, size=(h, w), mode='nearest')
                        warp_mask = torch.squeeze(warp_mask, 1)

                    warp_mask = warp_mask.type(torch.int64)
                    scores = F.cross_entropy(logits, warp_mask, reduction='none')
                    ret['consist_score'] = torch.mean(scores, dim=(1, 2))

                ret['warp_mask'] = ret['warp_mask'].type(torch.int32)

            # Project the PoI to the frame:
            if project_poi:
                poi = self.transform_poi(theta, self.court_poi)
                ret['poi'] = poi

        return ret