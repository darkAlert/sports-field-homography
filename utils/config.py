import argparse
import yaml


def parse_config(path_to_yaml):
    '''
    Parse config from an yaml-file
    '''
    config = None

    try:
        with open(path_to_yaml, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file:', path_to_yaml)

    return config

def make_base_parser():
    '''
    Create a list of reconstructor arguments
    '''
    parser = argparse.ArgumentParser(description='Reconstructor')
    parser.add_argument('--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('--conf_path', '-c', dest='conf_path', type=str, default=None,
                        help='Load config from a .yaml file')
    parser.add_argument('--viz', action='store_true', default=False,
                        help="Visualize the images as they are processed")
    parser.add_argument('--batchsize', '-bs', dest='batchsize', type=int, default=8,
                        help='Batch size')

    # Data:
    parser.add_argument('--img_dir', dest='img_dir', type=str, default=None,
                        help='Path to dir containing training images')
    parser.add_argument('--court_img', dest='court_img', type=str,
                        default='./assets/mask_ncaa_v4_nc4_m_onehot.png',
                        help='Path to court template image that will be projected by affine matrix')
    parser.add_argument('--court_poi', dest='court_poi', type=str,
                        default='./assets/template_ncaa_v4_points.json',
                        help='Path to court points of interest. Using in reprojection error')

    # Resolutions:
    parser.add_argument('--target_size', dest='target_size', default=(640,360),
                        help='Size of the input/output data')
    parser.add_argument('--unet_size', dest='unet_size', default=(640, 360),
                        help='Size of the UNET input/output')
    parser.add_argument('--warp_size', dest='warp_size', default=(640, 360),
                        help='Output size of warper')
    parser.add_argument('--court_size', dest='court_size', default=(640, 360),
                        help='Size of the court image template')

    # Segmentation (UNET):
    parser.add_argument('--use_unet', action='store_true', default=True,
                        help="Whether to use UNET or not")
    parser.add_argument('--unet_bilinear', action='store_true', default=False,
                        help="Use bilinear interpolation (True) or deconvolution (False) layers")
    parser.add_argument('--mask_classes', dest='mask_classes', type=int, default=4,
                        help='Number of segmentation mask classes')

    # Regression (ResNetSTN):
    parser.add_argument('--use_resnet', action='store_true', default=True,
                        help="Whether to use ResNetSTN or not")
    parser.add_argument('--resnet_name', type=str, default='resnet34',
                        help='Specify ResNetSTN model (resnet18, resnet34, resnet50, etc.)')
    parser.add_argument('--resnet_input', type=str, default='resnet34',
                        help='Specify type of input data. Can be \'img / mask / img+mask\'')
    parser.add_argument('--use_warper', action='store_true', default=True,
                        help="Whether to warp the court mask with homography or not")

    return parser


def get_training_args(ret_parser=False):
    '''
    Create a list of the Reconstructor arguments for training
    '''
    # Base parser:
    parser = make_base_parser()

    # Additional ResNetSTN args:
    parser.add_argument('--resnet_pretrained', type=str, default=None,
                        help='Whether to load ResNetSTN from a pretrained weights or not')

    # Training data:
    parser.add_argument('--mask_dir', dest='mask_dir', type=str, default=None,
                        help='Path to dir containing masks for given images')
    parser.add_argument('--anno_dir', dest='anno_dir', type=str, default=None,
                        help='Path to dir containing annotations for given images')
    parser.add_argument('--anno_keys', dest='anno_keys', type=str, default=None,
                        help='List of annotation keys that will be used as input data')
    parser.add_argument('--val_names', dest='val_names', type=str, default=None,
                        help='List of video names that will be used in validation step')
    parser.add_argument('--aug', dest='aug', type=str, default=None,
                        help='Augmentation')
    parser.add_argument('--only_ncaam', action='store_true', default=False,
                        help="Use only NCAAM dataset for training")

    # Training args:
    parser.add_argument('--opt', dest='opt', type=str, default='RMSprop',
                        help='Optimizer for training')
    parser.add_argument('--epochs', dest='epochs', type=int, default=8,
                        help='Number of epochs')
    parser.add_argument('--lr', dest='lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-8,
                        help='Weight decay')
    parser.add_argument('--val_step_n', dest='val_step_n', type=int, default=None,
                        help='Validation at each step n')
    parser.add_argument('--cp_dir', dest='cp_dir', type=str, default=None,
                        help='Path for saving checkpoints')
    parser.add_argument('--log_dir', dest='log_dir', type=str, default=None,
                        help='Path for saving tensorboard logs')

    # Losses:
    parser.add_argument('--rec_loss', type=str, default='MSE',
                        help='Whether to use MSE or SmoothL1 as reconstruction loss')
    parser.add_argument('--seg_loss', type=str, default='CE',
                        help='Segmentation loss. Can be \'CE\' (Cross Entropy) or \'focal\' (Focal loss)')
    parser.add_argument('--reproj_loss', type=str, default=None,
                        help='Whether to use Reprojection loss or not. Can be \'RRMSE\' or None')
    parser.add_argument('--consist_loss', type=str, default=None,
                        help='Whether to use Consistency loss or not. Can be CE/focal or None')
    parser.add_argument('--consist_start_iter', type=int, default=0,
                        help='The iteration number when the consistency loss starts applying')
    parser.add_argument('--seg_lambda', type=float, default=1.0,
                        help='Weighting factor for segmentation loss')
    parser.add_argument('--rec_lambda', type=float, default=10.0,
                        help='Weighting factor for reconstruction loss')
    parser.add_argument('--reproj_lambda', type=float, default=1.0,
                        help='Weighting factor for Reprojection loss')
    parser.add_argument('--consist_lambda', type=float, default=1.0,
                        help='Weighting factor for Consistency loss')

    return parser.parse_args() if ret_parser == False else parser


def get_prediction_args():
    '''
    Create a list of the Reconstructor arguments to make predictions
    '''
    # Base parser:
    parser = make_base_parser()

    # Prediction args:
    parser.add_argument('--video_path', type=str, default=None,
                        help='Path to video to process. If img_dir is empy, the video will be used')
    parser.add_argument('--dst_dir', type=str, default=None,
                        help='Directory where the results will be saved')
    parser.add_argument('--req_outputs', type=str, default='segm_mask,warp_mask,theta,poi,consistency,debug',
                        help='Output names to be computed and saved')
    parser.add_argument('--out_size', default=(1280, 720), nargs='+', type=int,
                        help='Output images size')
    parser.add_argument('--mask_type', type=str, default='gray',
                        help='Output mask type. Can be [bin / gray / rgb]')
    parser.add_argument('--mask_save_format', type=str, default='pickle',
                        help='Mask save format. Can be [png / pickle]')

    return parser.parse_args()


def get_test_args():
    parser = get_training_args(ret_parser=True)
    parser.description = 'Test'
    parser.add_argument('--test_epochs', dest='test_epochs', type=str, default=None,
                        help='List of epochs to test, e.g. 1,2,5')
    parser.add_argument('--metric_img_size', '-mis', dest='metric_img_size', default=(640, 360),
                        help='Metric image size')

    return parser.parse_args()

def replace_args(args, conf, ignore_keys=None):
    '''
    Replace reconstructor arguments with ones from parsed yaml config
    '''
    assert args is not None
    assert conf is not None

    if ignore_keys is None:
        ignore_keys = []

    for k in vars(args).keys():
        if k not in ignore_keys and k in conf:
            setattr(args, k, conf[k])

    return args

