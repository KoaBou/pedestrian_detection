import torch
import numpy as np
from shapely.geometry import Polygon
import json

import torch.nn.functional as F

def convert_format(boxes_array):
    """

    :param array: an array of shape [# bboxs, 4, 2]
    :return: a shapely.geometry.Polygon object
    """

    polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in boxes_array]
    return np.array(polygons)

def compute_iou(box, boxes):
    """Calculates IoU of the given box with the array of the given boxes.
    box: a polygon
    boxes: a vector of polygons
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    iou = [box.intersection(b).area / box.union(b).area for b in boxes]

    return np.array(iou, dtype=np.float32)

def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum suppression and returns indices of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.

    return an numpy array of the positions of picks
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    polygons = convert_format(boxes)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(polygons[i], polygons[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)

    return np.array(pick, dtype=np.int32)



def filter_pred(pred, config, out_size_factor, thres, nms_thres = None):
    geom = config["geometry"]

    offset_pred = pred["offset"].squeeze().detach()
    size_pred = pred["size"].squeeze().detach()
    yaw_pred = pred["yaw"].squeeze().detach()
    cls_pred = pred["cls"].squeeze().detach()

    cos_t, sin_t = torch.chunk(yaw_pred, 2, dim = 0)
    dx, dy = torch.chunk(offset_pred, 2, dim = 0)
    log_w, log_l = torch.chunk(size_pred, 2, dim = 0)

    cls_probs, cls_ids = torch.max(cls_pred, dim = 0)

    output_shape = [int((geom["y_max"] - geom["y_min"]) / geom["y_res"] / out_size_factor), 
                    int((geom["x_max"] - geom["x_min"]) / geom["x_res"] / out_size_factor)]

    y = torch.arange(output_shape[0])
    x = torch.arange(output_shape[1])

    xx, yy = torch.meshgrid(x, y, indexing="xy")
    xx = xx.to(offset_pred.device)
    yy = yy.to(offset_pred.device)

    center_y = dy + yy *  geom["y_res"] * out_size_factor + geom["y_min"]
    center_x = dx + xx *  geom["x_res"] * out_size_factor + geom["x_min"]
    center_x = center_x.squeeze()
    center_y = center_y.squeeze()
    l = torch.exp(log_l).squeeze()
    w = torch.exp(log_w).squeeze()
    yaw2 = torch.atan2(sin_t, cos_t).squeeze()
    yaw = yaw2 / 2

    if nms_thres is None:
        pooled = F.max_pool2d(cls_probs.unsqueeze(0), 3, 1, 1).squeeze()
        selected_idxs = torch.logical_and(cls_probs == pooled, cls_probs > thres)
    else:
        idxs = torch.logical_and(cls_probs > thres, cls_ids != 0)
        cls_ids = cls_ids[idxs]
        cls_probs = cls_probs[idxs]

        cos_t = torch.cos(yaw)
        sin_t = torch.sin(yaw)

        rear_left_x = center_x - l/2 * cos_t - w/2 * sin_t
        rear_left_y = center_y - l/2 * sin_t + w/2 * cos_t
        rear_right_x = center_x - l/2 * cos_t + w/2 * sin_t
        rear_right_y = center_y - l/2 * sin_t - w/2 * cos_t
        front_right_x = center_x + l/2 * cos_t + w/2 * sin_t
        front_right_y = center_y + l/2 * sin_t - w/2 * cos_t
        front_left_x = center_x + l/2 * cos_t - w/2 * sin_t
        front_left_y = center_y + l/2 * sin_t + w/2 * cos_t


        decoded_reg = torch.cat([rear_left_x.unsqueeze(0), rear_left_y.unsqueeze(0), rear_right_x.unsqueeze(0), rear_right_y.unsqueeze(0),
                                front_right_x.unsqueeze(0), front_right_y.unsqueeze(0), front_left_x.unsqueeze(0), front_left_y.unsqueeze(0)], axis=0)
        
        decoded_reg = decoded_reg.permute(1, 2, 0)
        decoded_reg = decoded_reg[idxs]
        corners = np.reshape(decoded_reg.numpy(), (-1, 4, 2))
        selected_idxs = non_max_suppression(corners, cls_probs.numpy(), nms_thres)

        center_x = center_x[idxs]
        center_y = center_y[idxs]
        l = l[idxs]
        w = w[idxs]
        yaw = yaw[idxs]

        

    boxes = np.stack([cls_ids[selected_idxs].cpu().numpy(), 
                      cls_probs[selected_idxs].cpu().numpy(), 
                      center_x[selected_idxs].cpu().numpy(), 
                      center_y[selected_idxs].cpu().numpy(), 
                      l[selected_idxs].cpu().numpy(), 
                      w[selected_idxs].cpu().numpy(), 
                      yaw[selected_idxs].cpu().numpy()])

    boxes = np.swapaxes(boxes, 0, 1)

    return boxes





