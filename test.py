import os
import torch
from torch.utils.data import DataLoader
from models import Reconstructor
from eval import eval_reconstructor
from utils.dataset import BasicDataset, split_on_train_val, open_court_template, open_court_poi
from utils.config import parse_config, get_test_args, replace_args
from utils.logger import get_logger


def test(args):
    '''
    Test the model
    '''
    conf_path = os.path.join(os.path.dirname(args.load), 'conf.yaml')
    assert (os.path.isfile(conf_path))

    # Read params and replace them with ones from yaml config:
    print ('Reading params from {}...'.format(conf_path))
    conf = parse_config(conf_path)
    ignore_keys = ['img_dir', 'mask_dir', 'anno_dir', 'batchsize', 'load', 'court_img', 'court_poi']
    args = replace_args(args, conf, ignore_keys=ignore_keys)

    args.resnet_pretrained = None
    args.anno_keys = ['poi']
    args.log_path = os.path.join(os.path.dirname(args.load), 'test_scores.txt')

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
    net.load_state_dict(torch.load(args.load, map_location=device))

    # Prepare dataset:
    test_ids, _ = split_on_train_val(args.img_dir, val_names=[])
    test_data = BasicDataset(test_ids, args.img_dir, args.mask_dir, args.anno_dir, args.anno_keys, args.mask_classes, args.target_size)
    data_loader = DataLoader(test_data, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    n_test = len(test_data)

    logger.info(f'''Starting testing:
            Model file:      {args.load}
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
            Metric img size: {args.metric_img_size}
        ''')

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Evaluate:
    start.record()
    result = eval_reconstructor(net, data_loader, device, args.metric_img_size, use_per_sample_weights=False)
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
    args = get_test_args()
    epochs = args.test_epochs.split(',')

    for e in epochs:
        cp_name = 'CP_epoch{}.pth'.format(e)
        args.load = os.path.join(args.cp_dir, cp_name)
        if not os.path.exists(args.load):
            print('Model file not found: {}'.format(args.load))
            continue
        test(args)