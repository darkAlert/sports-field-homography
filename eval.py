import torch
import torch.nn.functional as F
from models.dice_loss import dice_coeff
from models.losses import reprojection_loss, per_sample_weighted_criterion


def eval_net(net, loader, device, verbose=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    imgs, mask_pred = None, None

    print ('\nStarting validation...\n')

    for batch in loader:
        imgs, true_masks = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)

        with torch.no_grad():
            mask_pred = net(imgs)

        if net.n_classes > 1:
            tot += F.cross_entropy(mask_pred, true_masks).item()
        else:
            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()
            tot += dice_coeff(pred, true_masks).item()

    net.train()

    result = {'val_score': tot/n_val}
    if verbose:
        result['imgs'] = imgs.cpu()
        result['preds'] = mask_pred.cpu()

    return result


def eval_stn(net, loader, device, verbose=False):
    """Evaluation UNET+STN"""
    print('\nStarting validation...\n')
    ce_score, mse_score = 0, 0
    imgs, mask_pred, projected_masks = None, None, None
    mask_type = torch.long
    n_val = len(loader)

    net.eval()

    for batch in loader:
        imgs, true_masks = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)

        with torch.no_grad():
            mask_pred, projected_masks = net(imgs)

        # Scores:
        ce_score += F.cross_entropy(mask_pred, true_masks).item()
        gt_masks = true_masks.to(dtype=torch.float32) / float(net.n_classes)
        mse_score += F.mse_loss(projected_masks, gt_masks).item()

    net.train()

    result = {'val_tot_score': (ce_score+mse_score)/n_val,
              'val_ce_score': ce_score/n_val,
              'val_mse_score': mse_score/n_val}
    if verbose:
        result['imgs'] = imgs.cpu()
        result['preds'] = mask_pred.cpu()
        result['projs'] = projected_masks.cpu()

    return result


def vizualize_poi(theta, court_poi, imgs):
    '''Project the court PoI via the predicted homography'''
    import cv2
    import numpy as np

    h, w = imgs.shape[2], imgs.shape[3]
    np_imgs = imgs.cpu().numpy().astype('float32')
    np_imgs = np.transpose(np_imgs, (0, 2, 3, 1))
    np_imgs = (np_imgs * 255.0).astype('uint8')

    # Transform the points:
    theta_inv = torch.inverse(theta)
    trans_poi = transform_points(theta_inv, court_poi)
    np_trans_poi = trans_poi.cpu().numpy().astype('float32')

    for i, (out_img, out_poi) in enumerate(zip(np_imgs, np_trans_poi)):
        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        for pts in out_poi:
            x, y = int(round((pts[0] / 2.0 + 0.5) * w)), int(round((pts[1] / 2.0 + 0.5) * h))
            out_img = cv2.circle(out_img, (x, y), 3, (255, 0, 0), 2)
        out_path = '/media/darkalert/c02b53af-522d-40c5-b824-80dfb9a11dbb/boost/test/' + str(i) + '.png'
        cv2.imwrite(out_path, out_img)


    # # TRANSFORMATION BY HANDS #
    # theta_inv = torch.inverse(theta)
    # court_poi = torch.transpose(net.court_poi, 1, 2)
    #
    # proj_poi = torch.matmul(theta_inv, court_poi)
    # proj_poi[:, 0] = (proj_poi[:, 0] / proj_poi[:, 2])
    # proj_poi[:, 1] = (proj_poi[:, 1] / proj_poi[:, 2])
    #
    # proj_poi = torch.transpose(proj_poi, 1, 2)
    # poi = proj_poi.cpu().numpy().astype('float32')
    #
    # for i, (out_img, out_poi) in enumerate(zip(np_imgs, poi)):
    #     out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    #     for pts in out_poi:
    #         x, y = int(round((pts[0] / 2.0 + 0.5) * w)), int(round((pts[1] / 2.0 + 0.5) * h))
    #         out_img = cv2.circle(out_img, (x, y), 3, (0, 255, 0), 2)
    #     out_path = '/media/darkalert/c02b53af-522d-40c5-b824-80dfb9a11dbb/boost/test/' + str(i) + '.png'
    #     cv2.imwrite(out_path, out_img)


    # # TRANSFORMATION VIA cv2.perspectiveTransform #
    # np_theta = theta.cpu().numpy().astype('float32')[0]
    # np_theta = np.linalg.inv(np_theta)
    # np_poi = court_poi.cpu().numpy().astype('float32')[0]
    # np_poi = np.array([np_poi[:, 0:2]])
    #
    # projected_poi = cv2.perspectiveTransform(np_poi, np_theta)
    #
    # for i, (out_img, out_poi) in enumerate(zip(np_imgs, projected_poi)):
    #     out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    #     for pts in out_poi:
    #         # x,y = int(round(pts[0]*w)), int(round(pts[1]*h))
    #         # out_img = cv2.circle(out_img, (x, y), 3, (0, 255, 255), 2)
    #         x, y = int(round((pts[0] / 2.0 + 0.5) * w)), int(round((pts[1] / 2.0 + 0.5) * h))
    #         out_img = cv2.circle(out_img, (x, y), 3, (255, 255, 0), 2)
    #     out_path = '/media/darkalert/c02b53af-522d-40c5-b824-80dfb9a11dbb/boost/test/' + str(i) + '_.png'
    #     cv2.imwrite(out_path, out_img)



def eval_reconstructor(net, loader, device, target_size, use_per_sample_weights=True):
    """Evaluation UNet+ResNetReg"""
    ce_score, rec_score, reproj_score, reproj_px, consist_score = 0, 0, 0, 0, 0
    imgs, logits, rec_masks = None, None, None
    mask_type = torch.long
    n_val = len(loader)
    target_w, target_h = target_size[0], target_size[1]

    net.eval()
    counter = 0

    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device=device, dtype=torch.float32)
            gt_masks_i = batch['mask'].to(device=device, dtype=mask_type)
            gt_masks_f = gt_masks_i.to(dtype=torch.float32) / float(net.mask_classes)
            gt_poi, nonzeros, num_nonzero = None, None, None
            if 'poi' in batch:
                gt_poi = batch['poi'].to(device=device, dtype=torch.float32)
                nonzeros= batch['nonzeros'].to(device=device, dtype=torch.float32)
                num_nonzero = batch['num_nonzero'].to(device=device, dtype=torch.float32)
            counter += imgs.shape[0]

            preds = net(imgs)
            logits, theta, poi, warp_mask = None, None, None, None
            if 'logits' in preds:
                logits = preds['logits']
            if 'theta' in preds:
                theta = preds['theta']
            if 'poi' in preds:
                poi = preds['poi']
            if 'warp_mask' in preds:
                warp_masks = preds['warp_mask']

            # Scores:
            if use_per_sample_weights:
                gt_weights = batch['weight'].to(device=device)
                if logits is not None:
                    params = [F.cross_entropy, logits, gt_masks_i, gt_weights]
                    ce_score += per_sample_weighted_criterion(*params).item()
                if warp_masks is not None:
                    params = [F.mse_loss, warp_masks, gt_masks_f, gt_weights]
                    rec_score += per_sample_weighted_criterion(*params).item()
            else:
                if logits is not None:
                    ce_score += F.cross_entropy(logits, gt_masks_i).item()
                if warp_masks is not None:
                    rec_score += F.mse_loss(warp_masks, gt_masks_f).item()

            # Calculate the Consistency error beteween the predicted mask and the projected:
            if logits is not None and warp_masks is not None:
                warp_masks_i = (warp_masks * net.mask_classes).to(dtype=torch.long)
                consist_score += F.cross_entropy(logits, warp_masks_i).item()

            # Calculate the Reprojection Mean Squared Error (RRMSE):
            if gt_poi is not None and poi is not None:
                reproj_score += reprojection_loss(poi, gt_poi, nonzeros, num_nonzero, 'sum').item()

                # Normalize the points to the frame size:
                gt_poi[:, :, 0] = gt_poi[:, :, 0] * target_w
                gt_poi[:, :, 1] = gt_poi[:, :, 1] * target_h
                poi[:, :, 0] = poi[:, :, 0] * target_w
                poi[:, :, 1] = poi[:, :, 1] * target_h

                reproj_px += reprojection_loss(poi, gt_poi, nonzeros, num_nonzero, 'sum').item()

    net.train()

    result = {'val_seg_score': ce_score/n_val,
              'val_rec_score': rec_score/n_val,
              'val_reproj_score': reproj_score/counter,
              'val_reproj_px': reproj_px/counter,
              'val_consist_score': consist_score/n_val}
    result['imgs'] = imgs.cpu()
    if logits is not None:
        result['logits'] = logits.cpu()
    if warp_masks is not None:
        result['warp_masks'] = warp_masks.cpu()
    # vizualize_poi(theta, net.court_poi, imgs)

    return result