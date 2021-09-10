import os
import torch
from torch.utils.data import DataLoader
from models import Reconstructor
from eval import eval_reconstructor
from utils.dataset import BasicDataset, split_on_train_val, open_court_template, open_court_poi
from utils.config import parse_config, get_training_args, replace_args
from utils.logger import get_logger


def test(model_path, batchsize=None, metric_img_size=None):
    '''
    Test the model
    '''
    conf_path = os.path.join(os.path.dirname(model_path), 'conf.yaml')
    if not os.path.isfile(conf_path):
        conf_path = None

    # Read params and replace them with ones from yaml config:
    args = get_training_args()
    if conf_path is not None:
        print ('Reading params from {}...'.format(conf_path))
        conf = parse_config(conf_path)
        args = replace_args(args, conf)
    else:
        # Or set params manually:
        args.unet_bilinear = False
        args.mask_classes = 4
        args.resnet_name = 'resnet34'
        args.resnet_input = 'img+mask'
        args.target_size = [640, 360]
        args.segm_size = [640, 360]
        args.unet_size = [640, 360]

    args.court_img = './assets/pitch_mask_nc4_hd_onehot.png'
    args.court_poi = './assets/template_pitch_points.json'
    args.img_dir = '/media/darkalert/c02b53af-522d-40c5-b824-80dfb9a11dbb/sota/football/datasets/sota-pitch-test/frames/'
    args.mask_dir = '/media/darkalert/c02b53af-522d-40c5-b824-80dfb9a11dbb/sota/football/datasets/sota-pitch-test/masks/'
    args.anno_dir = '/media/darkalert/c02b53af-522d-40c5-b824-80dfb9a11dbb/sota/football/datasets/sota-pitch-test/anno/'
    args.anno_keys = ['poi']
    args.log_path = os.path.join(os.path.dirname(model_path), 'test_scores.txt')
    args.resnet_pretrained = None
    if batchsize is not None:
        args.batchsize = batchsize
    if metric_img_size is None:
        metric_img_size = args.target_size

    # Log:
    logger = get_logger(args.log_path, format='%(message)s')

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
                        resnet_pretrained = args.resnet_pretrained,
                        use_warper = args.use_warper,
                        warp_size = args.warp_size,
                        warp_with_nearest = True)
    net.to(device=device)
    net.load_state_dict(torch.load(model_path, map_location=device))

    # Prepare dataset:
    test_ids, _ = split_on_train_val(args.img_dir, val_names=[])
    test_data = BasicDataset(test_ids, args.img_dir, args.mask_dir, args.anno_dir, args.anno_keys, args.mask_classes, args.target_size)
    data_loader = DataLoader(test_data, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    n_test = len(test_data)

    logger.info(f'''Starting testing:
            Model file:      {model_path}
            Images dir:      {args.img_dir}
            Masks dir:       {args.mask_dir}
            Annotation dir:  {args.anno_dir}
            Annotation keys: {args.anno_keys}
            Logs file:       {args.log_path}
            Batch size:      {args.batchsize}
            Test size:       {n_test}
            Device:          {device.type}
            Target size:     {args.target_size}
            UNET input size: {args.unet_size}
            Bilinear:        {args.unet_bilinear}
            Mask classes:    {args.mask_classes}
            ResNetSTN:       {args.resnet_name}
            Resnet Input:    {args.resnet_input}
            Metric img size: {metric_img_size}
        ''')

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Evaluate:
    start.record()
    result = eval_reconstructor(net, data_loader, device, metric_img_size, use_per_sample_weights=False)
    end.record()
    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(end)
    reproj_px = result['val_reproj_px']
    reproj_score = result['val_reproj_score']
    seg_score = result['val_seg_score']
    rec_score = result['val_rec_score']

    logger.info(f'''Test scores:
            Reprojection px:     {reproj_px}
            Reprojection RMSE:   {reproj_score}
            Segmentation CE:     {seg_score}
            Reconstruction MSE:  {rec_score}
            Elapsed msec:        {elapsed_time}
        ''')

    # Free memory:
    del net, data_loader, test_data, logger
    torch.cuda.empty_cache()

    print ('All done!')


if __name__ == "__main__":
    names = []
    # names.append('sota-pitch-v2-640x360-aug_unet-resnet34-deconv-mask_ce-l1-rrmse-focal_pre')
    names.append('sota-pitch-v2-640x360-aug_unet-resnet34-deconv-img+mask_ce-l1-rrmse-focal_pre')
    epochs = [3, 9, 10, 13, 14, 15]
    batchsize = 20

    for name in names:
        for e in epochs:
            model_path = '/home/darkalert/builds/sports-field-homography/checkpoints/pitch/{}/CP_epoch{}.pth'.format(name, e)
            if not os.path.exists(model_path):
                print('Model file not found: {}'.format(model_path))
                continue
            test(model_path, batchsize=batchsize, metric_img_size=(640,360))