import os
import sys
from tqdm import tqdm
import signal
import logging
from shutil import copyfile

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset, split_on_train_val, worker_init_fn, open_court_template, open_court_poi
from torch.utils.data import DataLoader
import kornia

from eval import eval_reconstructor
from models.reconstructor import Reconstructor
from models.losses import ReprojectionLoss, per_sample_weighted_criterion
from utils.postprocess import preds_to_masks, onehot_to_image
from utils.config import parse_config, get_training_args, replace_args
from utils.logger import get_logger


def prepare_dataloader(img_dir, mask_dir, anno_dir, anno_keys, val_names,
                       mask_classes, use_uv, batch_size, target_size, aug, only_ncaam):
    '''
    Prepare DataLoaders for train and validation sets
    '''
    train_ids, val_ids = split_on_train_val(img_dir, val_names, only_ncaam=only_ncaam)
    train = BasicDataset(train_ids, img_dir, mask_dir, anno_dir, anno_keys, mask_classes, use_uv, target_size, aug=aug)
    val = BasicDataset(val_ids, img_dir, mask_dir, anno_dir, anno_keys, mask_classes, use_uv, target_size)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8,
                              pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8,
                            pin_memory=True, drop_last=True)
    n_train = len(train)
    n_val = len(val)

    return train_loader, n_train, val_loader, n_val


def train_net(net, device, train_loader, n_train, val_loader, batch_size, val_step_n,
              seg_loss, seg_lambda, rec_loss, rec_lambda, reproj_loss, reproj_lambda,
              consist_loss, consist_lambda, consist_start_iter,
              uv_loss=None, uv_lambda=None,
              opt='RMSprop', epochs=5, lr=0.0001, w_decay=1e-8,
              target_size=(1280,720),
              cp_dir=None, log_dir=None,
              logger=None, vizualize=False):
    '''
    Train the Reconstructor model
    '''
    if logger is None:
        logger = logging

    val_step_n = val_step_n if val_step_n is not None else int(n_train / batch_size) + 1

    logger.info(f'''# Starting training:
            Optimizer:       {opt}
            Epochs:          {epochs}
            Val step:        {val_step_n}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Weight decay:    {w_decay}
            Segmentation:    {seg_loss}
            Reconstruction:  {rec_loss}
            Reprojection:    {reproj_loss}
            UV:              {uv_loss}
            Consistency:     {consist_loss}
            Cons start iter: {consist_start_iter}
            Seg Lambda:      {seg_lambda}
            Rec Lambda:      {rec_lambda}
            Reproj Lambda:   {reproj_lambda}
            UV Lambda:       {uv_lambda}
            Consist Lambda:  {consist_lambda}
            Checkpoints dir: {cp_dir}
            Log dir:         {log_dir}
            Device:          {device.type}
            Vizualize:       {vizualize}
    ''')

    writer = SummaryWriter(log_dir=log_dir,
                           comment=f'LR_{lr}_BS_{batch_size}_SIZE_{target_size}')

    # Oprimizer:
    if opt == 'RMSprop':
        optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=w_decay, momentum=0.9)
    elif opt == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=w_decay, momentum=0.9)
    elif opt == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=w_decay)
    else:
        print ('optimizer {} does not support yet'.format(opt))
        raise NotImplementedError

    # Scheduler:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    # Segmentation loss:
    seg_criterion = None
    if seg_loss is not None:
        if seg_loss == 'CE':
            seg_criterion = nn.CrossEntropyLoss(reduction='none')
        elif seg_loss == 'focal':
            seg_criterion = kornia.losses.FocalLoss(alpha=1.0, gamma=2.0, reduction='none')
        else:
            raise NotImplementedError

    # Reconstruction loss:
    rec_criterion = None
    if rec_loss is not None:
        if rec_loss == 'MSE':
            rec_criterion = nn.MSELoss(reduction='none')
        elif rec_loss == 'SmoothL1':
            rec_criterion = nn.SmoothL1Loss(reduction='none')
        else:
            raise NotImplementedError

    # Reprojection loss:
    reproj_creiterion = None
    if reproj_loss is not None:
        if reproj_loss == 'RRMSE':
            reproj_creiterion = ReprojectionLoss()
        else:
            raise NotImplementedError

    # Consistency loss:
    consistency_creiterion = None
    if consist_loss is not None:
        if consist_loss == 'CE':
            consistency_creiterion = nn.CrossEntropyLoss()
        elif consist_loss == 'focal':
            consistency_creiterion = kornia.losses.FocalLoss(alpha=1.0, gamma=2.0, reduction='mean')

    # UV loss:
    uv_criterion = None
    if net.unet_uv and uv_loss is not None:
        if uv_loss == 'MSE':
            uv_criterion = nn.MSELoss(reduction='none')
        elif uv_loss == 'SmoothL1':
            uv_criterion = nn.SmoothL1Loss(reduction='none')
        else:
            raise NotImplementedError

    num_classes = net.mask_classes
    global_step = 0

    # Training loop:
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                # Get data:
                imgs = batch['image'].to(device=device, dtype=torch.float32)
                gt_masks = batch['mask'].to(device=device)
                gt_weights = batch['weight'].to(device=device)
                gt_uv = None
                if net.unet_uv:
                    gt_uv = batch['uv'].to(device=device)

                assert imgs.shape[1] == 3, \
                    f'Network has been defined with 3 input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                # Forward:
                preds = net(imgs)
                logits, theta, poi, warp_mask, uv = None, None, None, None, None
                if 'logits' in preds:
                    logits = preds['logits']
                if 'theta' in preds:
                    theta = preds['theta']
                if 'poi' in preds:
                    poi = preds['poi']
                if 'warp_mask' in preds:
                    warp_mask = preds['warp_mask']
                if 'uv' in preds:
                    uv = preds['uv']

                loss = torch.zeros(1, device=device, dtype=torch.float32)
                logs = {}

                # Caluclate CrossEntropy loss:
                if seg_criterion is not None:
                    seg_loss = per_sample_weighted_criterion(seg_criterion, logits, gt_masks, gt_weights) * seg_lambda
                    loss += seg_loss
                    writer.add_scalar('Loss/train seg', seg_loss.item(), global_step)
                    logs['Seg_loss'] = seg_loss.item()

                # Calculate Reconstruction loss for homography:
                if rec_criterion is not None:
                    gt_masks_f = gt_masks.to(dtype=torch.float32) / float(num_classes)
                    rec_loss = per_sample_weighted_criterion(rec_criterion, warp_mask, gt_masks_f, gt_weights) * rec_lambda
                    loss += rec_loss
                    writer.add_scalar('Loss/train rec', rec_loss.item(), global_step)
                    logs['Rec_loss'] = rec_loss.item()

                # Calculate UV loss:
                if uv_criterion is not None:
                    uv_loss = per_sample_weighted_criterion(uv_criterion, uv, gt_uv, gt_weights) * uv_lambda
                    loss += uv_loss
                    writer.add_scalar('Loss/train uv', uv_loss.item(), global_step)
                    logs['UV_loss'] = uv_loss.item()

                # Calculate Reprojection loss for homography:
                if reproj_creiterion is not None:
                    gt_poi = batch['poi'].to(device=device, dtype=torch.float32)
                    nonzeros = batch['nonzeros'].to(device=device, dtype=torch.float32)
                    num_nonzero = batch['num_nonzero'].to(device=device, dtype=torch.float32)
                    reproj_loss = reproj_creiterion(poi, gt_poi, nonzeros, num_nonzero) * reproj_lambda
                    loss += reproj_loss
                    writer.add_scalar('Loss/train reproj', reproj_loss.item(), global_step)
                    logs['Reproj_loss'] = reproj_loss.item()

                # Calculate Consistency loss between the predicted mask and the projected:
                if consistency_creiterion is not None \
                        and global_step*batch_size >= consist_start_iter:
                    rec_masks_int = (warp_mask * num_classes).to(dtype=torch.long)
                    consist_loss = consistency_creiterion(logits, rec_masks_int) * consist_lambda
                    loss += consist_loss
                    writer.add_scalar('Loss/train consistency', consist_loss.item(), global_step)
                    logs['Cons_loss'] = consist_loss.item()

                # Log:
                writer.add_scalar('Loss/train', loss.item(), global_step)
                logs['Tot_loss'] = loss.item()
                pbar.set_postfix(**logs)

                # Backward:
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

                # Validation step:
                if global_step % val_step_n == 0:
                    print('\nStarting validation...')

                    for tag, value in net.named_parameters():
                        t = tag.replace('.', '/')
                        writer.add_histogram('weights/' + t, value.data.cpu().numpy(), global_step)
                        if value.grad is not None:
                            writer.add_histogram('grads/' + t, value.grad.data.cpu().numpy(), global_step)

                    # Evaluate:
                    result = eval_reconstructor(net, val_loader, device, target_size, True)
                    val_ce_score = result['val_seg_score']
                    val_rec_score = result['val_rec_score']
                    val_reproj_score = result['val_reproj_score']
                    val_reproj_px = result['val_reproj_px']
                    val_consist_score = result['val_consist_score']
                    val_uv_score = result['val_uv_score']
                    val_tot_score = val_ce_score + val_rec_score + val_reproj_score + val_consist_score + val_uv_score
                    scheduler.step(val_reproj_px)

                    # Validation log:
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                    writer.add_scalar('Loss/test', val_tot_score, global_step)
                    writer.add_scalar('Loss/test_seg', val_ce_score, global_step)
                    writer.add_scalar('Loss/test_rec', val_rec_score, global_step)
                    writer.add_scalar('Loss/test_uv', val_uv_score, global_step)
                    writer.add_scalar('Loss/test_reproj', val_reproj_px, global_step)
                    writer.add_scalar('Loss/test_consist', val_consist_score, global_step)
                    print ('\n')
                    logger.info('[Validation, epoch: {} of {}, step: {}] Tot: {}, seg: {}, rec: {}, '
                                'uv: {}, reproj: {}({:.3f})px, cons: {}'.
                                format(epoch + 1, epochs, global_step,
                                       val_tot_score, val_ce_score, val_rec_score, val_uv_score,
                                       val_reproj_score, val_reproj_px, val_consist_score))

                    if lr != optimizer.param_groups[0]['lr']:
                        lr = optimizer.param_groups[0]['lr']
                        logger.info('Learning rate has been changed: {}'.format(lr))

                    if vizualize:
                        # Postprocess predicted mask for tensorboard vizualization:
                        output = [result['imgs']]
                        if 'logits' in result:
                            pred_masks = preds_to_masks(result['logits'], net.mask_classes)
                            pred_masks = onehot_to_image(pred_masks, num_classes)
                            pred_masks = pred_masks[..., ::-1]          # rgb to bgr
                            pred_masks = np.transpose(pred_masks, (0, 3, 1, 2))
                            pred_masks = pred_masks.astype(np.float32) / 255.0
                            output.append(pred_masks)
                        if 'warp_masks' in result:
                            warp_masks = result['warp_masks'] * num_classes
                            warp_masks = warp_masks.type(torch.IntTensor).cpu().numpy().astype(np.uint8)
                            warp_masks = onehot_to_image(warp_masks, num_classes)
                            warp_masks = warp_masks[..., ::-1]
                            warp_masks = np.transpose(warp_masks, (0, 3, 1, 2))
                            warp_masks = warp_masks.astype(np.float32) / 255.0
                            output.append(warp_masks)
                        if 'uv_masks' in result:
                            uv_masks = result['uv_masks']
                            uv_masks = uv_masks.numpy().astype(np.float32)
                            # uv_masks = uv_masks[..., ::-1]
                            z = np.zeros((uv_masks.shape[0], 1, uv_masks.shape[-2], uv_masks.shape[-1]), dtype=np.float32)
                            uv_masks = np.concatenate((uv_masks, z), axis=1)
                            output.append(uv_masks)

                        # Concatenate all images:
                        output = np.concatenate(output, axis=2)

                        # Save the results for tensorboard vizualization:
                        writer.add_images('output', output, global_step)

        # Save checkpoint:
        if cp_dir is not None:
            try:
                os.mkdir(cp_dir)
                logger.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       cp_dir + f'CP_epoch{epoch + 1}.pth')
            logger.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


if __name__ == '__main__':
    '''
    Reconstructor network training
    '''
    # Read params and replace them with ones from yaml config:
    args = get_training_args()
    if args.conf_path is not None:
        conf = parse_config(args.conf_path)
        args = replace_args(args, conf)

    # Make logger:
    if not os.path.exists(args.cp_dir):
        os.makedirs(args.cp_dir)
    log_path = os.path.join(os.path.dirname(args.cp_dir), 'train.txt')
    logger = get_logger(log_path, format='%(message)s')

    # Choose device (CUDA or CPU):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the court template image and the court points of interest:
    court_img = open_court_template(args.court_img,
                                    num_classes=args.mask_classes,
                                    size=args.court_size,
                                    batch_size=args.batchsize)
    court_img = court_img.to(device=device)
    court_poi = open_court_poi(args.court_poi, args.batchsize)
    court_poi = court_poi.to(device=device)

    # Init Reconstructor:
    net = Reconstructor(court_img, court_poi,
                        target_size = args.target_size,
                        mask_classes = args.mask_classes,
                        use_unet = args.use_unet,
                        unet_bilinear = args.unet_bilinear,
                        unet_size = args.unet_size,
                        unet_uv = args.unet_uv,
                        use_resnet = args.use_resnet,
                        resnet_name = args.resnet_name,
                        resnet_input = args.resnet_input,
                        resnet_pretrained = args.resnet_pretrained,
                        use_warper = args.use_warper,
                        warp_size = args.warp_size)

    logger.info(f'''# Reconstructor network overview:
            Target size:      {args.target_size}
            Court img path:   {args.court_img}
            Court PoI path:   {args.court_poi}
            Court img size:   {args.court_size}
            ---UNet:
            Use UNet:         {args.use_unet}
            UNet bilinear:    {args.unet_bilinear}
            UNet size:        {args.unet_size}
            Mask classes:     {args.mask_classes}
            UNet UV:          {args.unet_uv}
            ---ResNetSTN:
            Use ResNet:       {args.use_resnet}
            ResNet name:      {args.resnet_name}
            ResNet input:     {args.resnet_input}
            ResNet weights:   {args.resnet_pretrained}
            Use warper:       {args.use_warper}
            Warp size:        {args.warp_size}
        ''')

    # Restore the model from a checkpoint:
    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logger.info(f'Model loaded from {args.load}\n')
    net.to(device=device)              # cudnn.benchmark = True: faster convolutions, but more memory

    # Prepare DataLoaders for train and val datasets:
    train_loader, n_train, val_loader, n_val =  prepare_dataloader(img_dir=args.img_dir,
                                                                   mask_dir=args.mask_dir,
                                                                   anno_dir=args.anno_dir,
                                                                   anno_keys=args.anno_keys,
                                                                   val_names=args.val_names,
                                                                   mask_classes=args.mask_classes,
                                                                   use_uv=args.unet_uv,
                                                                   batch_size=args.batchsize,
                                                                   target_size=args.target_size,
                                                                   aug=args.aug,
                                                                   only_ncaam=args.only_ncaam)
    logger.info(f'''# Dataset overview:
            Images dir:       {args.img_dir}
            Masks dir:        {args.mask_dir}
            Annotation dir:   {args.anno_dir}
            Annotation keys:  {args.anno_keys}
            Validation names: {args.val_names}
            Only NCAAM:       {args.only_ncaam}
            Augmentation:     {args.aug}
            Data resolution:  {args.target_size}
            Training size:    {n_train}
            Validation size:  {n_val}
          ''')

    # Copy config:
    if not os.path.isdir(args.cp_dir):
        os.mkdir(args.cp_dir)
    copyfile(args.conf_path, os.path.join(args.cp_dir,'conf.yaml'))

    # Check training params:
    if args.use_unet == False:
        args.seg_loss = None
        args.consist_loss = None
    if args.use_resnet == False:
        args.rec_loss = None
        args.reproj_loss = None
        args.consist_loss = None

    # Define save model function:
    def save_model(a1=None, a2=None):
        path = os.path.join(args.cp_dir, 'last.pth')
        torch.save(net.state_dict(), path)
        logger.info('Saved interrupt to {}'.format(path))
        sys.exit(0)
    signal.signal(signal.SIGTERM, save_model)

    # Run training:
    try:
        if not os.path.exists(args.cp_dir): os.makedirs(args.cp_dir)
        if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)

        train_net(net=net,
                  device=device,
                  train_loader=train_loader,
                  n_train=n_train,
                  val_loader=val_loader,
                  batch_size=args.batchsize,
                  val_step_n=args.val_step_n,
                  seg_loss=args.seg_loss,
                  seg_lambda = args.seg_lambda,
                  rec_loss=args.rec_loss,
                  rec_lambda=args.rec_lambda,
                  reproj_loss=args.reproj_loss,
                  reproj_lambda=args.reproj_lambda,
                  consist_loss=args.consist_loss,
                  consist_lambda=args.consist_lambda,
                  consist_start_iter = args.consist_start_iter,
                  uv_loss=args.uv_loss,
                  uv_lambda=args.uv_lambda,
                  opt=args.opt,
                  epochs=args.epochs,
                  lr=args.lr,
                  w_decay=args.weight_decay,
                  target_size=args.target_size,
                  cp_dir=args.cp_dir,
                  log_dir=args.log_dir,
                  logger=logger,
                  vizualize=args.viz)
    except KeyboardInterrupt:
        save_model()
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
