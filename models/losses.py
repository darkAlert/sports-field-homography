import types
import torch
import torch.nn as nn


def reprojection_loss(inputs, targets, nonzeros, num_nonzero, reduction='mean'):
    '''
    Calculate the distance between the input points and target ones
    '''
    dist = torch.sqrt(torch.sum(torch.pow(targets - inputs, 2), dim=2))
    loss = torch.sum(dist * nonzeros, dim=1) / num_nonzero

    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)

    return loss


class ReprojectionLoss(nn.Module):
    '''
    Reprojection Root Mean Squared Error loss
    '''
    def __init__(self):
        super(ReprojectionLoss, self).__init__()

    def forward(self, inputs, targets, nonzeros, num_nonzero, reduction='mean'):
        return reprojection_loss(inputs, targets, nonzeros, num_nonzero, reduction)



def per_sample_weighted_criterion(criterion, inputs, targets, per_sample_weights):
    if isinstance(criterion, types.FunctionType):
        loss = criterion(inputs, targets, reduction='none')
    else:
        loss = criterion(inputs, targets)
    loss = torch.mean(loss, dim=(1, 2)) * per_sample_weights
    loss = torch.mean(loss)

    return loss