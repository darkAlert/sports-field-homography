""" Full assembly of the parts to form the complete network """
from .unet_parts import *
import kornia


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNetSTN(nn.Module):
    def __init__(self, n_channels, n_classes, template, bilinear=True):
        super(UNetSTN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # Regressor for the 3 * 2 affine matrix:
        self.conv_reg = nn.Conv2d(1024 // factor, 8, kernel_size=1)
        self.reg = nn.Sequential(
            nn.Linear(8 * 22 * 40, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation:
        self.reg[-1].weight.data.zero_()
        self.reg[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # Court template that will be projected by affine matrix:
        self.template = template

    def stn(self, x, template):
        xl = self.conv_reg(x)
        xl = xl.view(xl.shape[0], -1)
        theta = self.reg(xl)
        theta = theta.view(-1, 2, 3)

        n = x.shape[0]
        t = template[0:n]

        grid = F.affine_grid(theta, t.size())
        proj = F.grid_sample(t, grid)

        return proj.squeeze(1)

    def forward(self, x):
        # UNet:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        # STN:
        proj = self.stn(x5, self.template)

        return logits, proj


class CourtReconstruction(nn.Module):
    '''
    The Court Reconstruction model consists of UNET, which learns
    to segment the court mask, and Spatial Transformer Network (STN),
    which learns to predict the homography matrix based on
    the predicted segmentation mask.
    '''
    def __init__(self, n_channels, n_classes, template, target_size, bilinear=True):
        super(CourtReconstruction, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # Regressor for the 3x3 transformation matrix:
        self.conv_reg = nn.Conv2d(1024 // factor, 8, kernel_size=1)
        self.reg = nn.Sequential(
            nn.Linear(8 * 22 * 40, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 3)
        )
        # Initialize the weights/bias with identity transformation:
        self.reg[-1].weight.data.zero_()
        self.reg[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float))

        # Court template that will be warped by the learnt transformation matrix:
        self.template = template

        # Warper:
        h, w = target_size[1], target_size[0]
        self.warper = kornia.HomographyWarper(h, w)#, mode='nearest')

    def stn(self, x, template):
        xl = self.conv_reg(x)
        xl = xl.view(xl.shape[0], -1)
        theta = self.reg(xl)
        theta = theta.view(-1, 1, 3, 3)

        # Determine batch size:
        bs = x.shape[0]
        template = template[0:bs]

        warped = self.warper(template, theta)

        return warped.squeeze(1)

    def forward(self, x):
        # UNet:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        # STN:
        proj = self.stn(x5, self.template)

        return logits, proj