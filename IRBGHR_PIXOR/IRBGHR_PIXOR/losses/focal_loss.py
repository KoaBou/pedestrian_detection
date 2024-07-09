import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_1.one_hot import one_hot

def modified_focal_loss(pred, gt):
    '''focal loss from CornerNet''' 
    # https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/losses.py
    # https://www.kaggle.com/code/kyoshioka47/centernet-starterkit-pytorch
    # https://www.programcreek.com/python/example/90318/tensorflow.pow

    #pred = torch.clamp(pred.sigmoid(), min = 1e-4, max = 1 - 1e-4)
    pred = pred.sigmoid()
    pos_inds = gt.eq(1) # sử dụng như boolean mask để xác định vị trí pos_inds inds(y=1) foreground gt = torch.tensor([0, 1, 1, 0, 1])
                        # tensor([False,  True,  True, False,  True])
    neg_inds = gt.lt(1) # sử dụng như boolean mask để xác định vị trí neg_inds inds(y=0) background gt = torch.tensor([0, 1, 1, 0, 1])
                        # tensor([True, False, False, True, False])

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    pos_pred = pred[pos_inds]
    neg_pred = pred[neg_inds]

    neg_pred[neg_pred == 1] = 1 - 1e-7
    pos_pred[pos_pred == 0] = 1e-7

    #print ("pos_pred",pos_pred)
    #print ("neg_pred",neg_pred)
    
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    
    #print ("pred",pred)
    #print ("gt",gt)
    #print ("loss",loss)
    

    return loss


# taken from https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py
# https://github.com/kuangliu/pytorch-retinanet/blob/master/loss.py
# https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py
def focal_loss(
    #https://arxiv.org/pdf/1708.02002v2.pdf  bài này chi tiết về focal_loss function
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 4.0,
    gamma: float = 2.0,
    reduction: str = 'mean',
    alphas = [1, 1, 1, 1, 1]
) -> torch.Tensor:

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(f'Expected target size {out_size}, got {target.size()}')

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1)
    log_input_soft: torch.Tensor = F.log_softmax(input, dim=1)
   
    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1.0, gamma)

    focal = -alpha * weight * log_input_soft
    #focal = -log_input_soft
    
    if int(alphas[0]) != 1:
        for i in range(len(alphas)):
            focal[:, i, ...] *= alphas[i] 
    
    #print(target_one_hot)
    loss_tmp = torch.einsum('bc...,bc...->b...', (target_one_hot, focal))

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


class FocalLoss(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.alpha: float = config["alpha"]
        self.gamma: float = config["gamma"]
        self.reduction: str = config["reduction"]


    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction)