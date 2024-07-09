import numpy as np
from torch.utils.data import Dataset
import torch
import os
import matplotlib.pyplot as plt
import math
import json
import time
import sys

import os
import sys
sys.path.append(os.getcwd())


from utils_1.preprocess import trasform_label2metric, get_points_in_a_rotated_box
from utils_1.transform import Random_Rotation, Random_Scaling, OneOf, Random_Translation
from utils_1.gaussian import gaussian_radius, draw_heatmap_gaussian

def trasform_label2metric(label, geometry, ratio=4):
    '''
    :param label: numpy array of shape [..., 2] of coordinates in label map space
    :return: numpy array of shape [..., 2] of the same coordinates in metric space
    '''
    
    metric = np.copy(label).astype(np.float32)
    metric[..., 0] = metric[..., 0] * ratio * geometry["x_res"] + geometry["x_min"]
    metric[..., 1] = metric[..., 1] * ratio * geometry["y_res"] + geometry["y_min"]

    return metric

def get_points_in_a_rotated_box(corners, label_shape=[200, 175]):
    def minY(x0, y0, x1, y1, x):
        if x0 == x1:
            # vertical line, y0 is lowest
            return int(math.floor(y0))

        m = (y1 - y0) / (x1 - x0)

        if m >= 0.0:
            # lowest point is at left edge of pixel column
            return int(math.floor(y0 + m * (x - x0)))
        else:
            # lowest point is at right edge of pixel column
            return int(math.floor(y0 + m * ((x + 1.0) - x0)))


    def maxY(x0, y0, x1, y1, x):
        if x0 == x1:
            # vertical line, y1 is highest
            return int(math.ceil(y1))

        m = (y1 - y0) / (x1 - x0)

        if m >= 0.0:
            # highest point is at right edge of pixel column
            return int(math.ceil(y0 + m * ((x + 1.0) - x0)))
        else:
            # highest point is at left edge of pixel column
            return int(math.ceil(y0 + m * (x - x0)))


    # view_bl, view_tl, view_tr, view_br are the corners of the rectangle
    view = [(corners[i, 0], corners[i, 1]) for i in range(4)]

    pixels = []

    # find l,r,t,b,m1,m2
    l, m1, m2, r = sorted(view, key=lambda p: (p[0], p[1]))
    b, t = sorted([m1, m2], key=lambda p: (p[1], p[0]))
    
    lx, ly = l
    rx, ry = r
    bx, by = b
    tx, ty = t
    m1x, m1y = m1
    m2x, m2y = m2

    xmin = 0
    ymin = 0
    xmax = label_shape[1]
    ymax = label_shape[0]

    # inward-rounded integer bounds
    # note that we're clamping the area of interest to (xmin,ymin)-(xmax,ymax)
    lxi = max(int(math.ceil(lx)), xmin)
    rxi = min(int(math.floor(rx)), xmax)
    byi = max(int(math.ceil(by)), ymin)
    tyi = min(int(math.floor(ty)), ymax)

    x1 = lxi
    x2 = rxi

    for x in range(x1, x2):
        xf = float(x)

        if xf < m1x:
            # Phase I: left to top and bottom
            y1 = minY(lx, ly, bx, by, xf)
            y2 = maxY(lx, ly, tx, ty, xf)

        elif xf < m2x:
            if m1y < m2y:
                # Phase IIa: left/bottom --> top/right
                y1 = minY(bx, by, rx, ry, xf)
                y2 = maxY(lx, ly, tx, ty, xf)

            else:
                # Phase IIb: left/top --> bottom/right
                y1 = minY(lx, ly, bx, by, xf)
                y2 = maxY(tx, ty, rx, ry, xf)

        else:
            # Phase III: bottom/top --> right
            y1 = minY(bx, by, rx, ry, xf)
            y2 = maxY(tx, ty, rx, ry, xf)

        y1 = max(y1, byi)
        y2 = min(y2, tyi)

        for y in range(y1, y2):
            pixels.append((x, y))

    return pixels

class Dataset(Dataset):
    def __init__(self, data_file, config, aug_config, cls_encoding, task = "train") -> None:
        self.data_file = data_file
        # stores fine names and data types in self.data_list and self.data_type_list
        self.create_data_list()
        self.config = config
        # depending on this task, we decide whether we want to load certain info (i.e. labels not available for testing sometimes)
        self.task = task
        # what kind of encoding we want to use for classification. Available options are : gaussian, inverse_distance, binary
        self.cls_encoding = cls_encoding
        self.num_classes = self.config["num_classes"]
        # add background to classes (0 will be background class)
        if cls_encoding == "binary":
            self.num_classes += 1

        self.transforms = self.get_transforms(aug_config)
        self.augment = OneOf(self.transforms, aug_config["p"])

        # downsample ratio
        self.out_size_factor = config["out_size_factor"]

        # calculate output shape
        geom = self.config[self.data_type_list[0]]["geometry"]
        self.output_shape = [int((geom["x_max"] - geom["x_min"]) / geom["x_res"] / self.out_size_factor), int((geom["y_max"] - geom["y_min"]) / geom["y_res"] / self.out_size_factor)]

        
       

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file = '{}.bin'.format(self.data_list[idx])
        data_type = self.data_type_list[idx]

        pointcloud_folder = os.path.join(self.config[data_type]["location"], "pointcloud")
        lidar_path = os.path.join(pointcloud_folder, file)

        points = self.read_points(lidar_path)

        if self.task == "test":
            scan = self.voxelize(points, self.config[data_type]["geometry"])
            scan = torch.from_numpy(scan)
            scan = scan.permute(2, 0, 1)
            return {"voxel": scan,
                    "points": points,
                    "dtype": data_type
                }

        boxes = self.get_boxes(idx)

        if self.task == "train" and boxes.shape[0] != 0:
            points, boxes[:, 1:] = self.augment(points, boxes[:, 1:8])

        boxes = self.filter_boxes(boxes, data_type)

        scan = self.voxelize(points, self.config[data_type]["geometry"])
        scan = torch.from_numpy(scan)
        scan = scan.permute(2, 0, 1)
        
        labels = self.get_label(boxes, self.config[data_type]["geometry"])
        labels["voxel"] = scan

        if self.task == "val":
            class_list, boxes = self.read_bbox(boxes)
            labels["cls_list"] = class_list
            labels["points"] = points
            labels["boxes"] = boxes
            labels["dtype"] = data_type  
            
            return labels
        
        return labels


    def read_points(self, lidar_path):
        return np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

    def voxelize(self, points, geometry):
        x_min = geometry["x_min"]
        x_max = geometry["x_max"]
        y_min = geometry["y_min"]
        y_max = geometry["y_max"]
        z_min = geometry["z_min"]
        z_max = geometry["z_max"]
        x_res = geometry["x_res"]
        y_res = geometry["y_res"]
        z_res = geometry["z_res"]

        x_size = int((x_max - x_min) / x_res)
        y_size = int((y_max - y_min) / y_res)
        z_size = int((z_max - z_min) / z_res)

        eps = 0.001

        #clip points
        x_indexes = np.logical_and(points[:, 0] > x_min + eps, points[:, 0] < x_max - eps)
        y_indexes = np.logical_and(points[:, 1] > y_min + eps, points[:, 1] < y_max - eps)
        z_indexes = np.logical_and(points[:, 2] > z_min + eps, points[:, 2] < z_max - eps)
        pts = points[np.logical_and(np.logical_and(x_indexes, y_indexes), z_indexes)]

        occupancy_mask = np.zeros((pts.shape[0], 3), dtype = np.int32)
        voxels = np.zeros((x_size, y_size, z_size), dtype = np.float32)
        occupancy_mask[:, 0] = (pts[:, 0] - x_min) // x_res
        occupancy_mask[:, 1] = (pts[:, 1] - y_min) // y_res
        occupancy_mask[:, 2] = (pts[:, 2] - z_min) // z_res

        idxs = np.array([occupancy_mask[:, 0].reshape(-1), occupancy_mask[:, 1].reshape(-1), occupancy_mask[:, 2].reshape( -1)])

        voxels[idxs[0], idxs[1], idxs[2]] = 1
        return np.swapaxes(voxels, 0, 1)


    def get_boxes(self, idx):
        '''
        :param i: the ith velodyne scan in the train/val set
        : return boxes of shape N:8

        '''

        f_name = '{}.txt'.format(self.data_list[idx])
        data_type = self.data_type_list[idx]
        label_folder = os.path.join(self.config[data_type]["location"], "label")
        label_path = os.path.join(label_folder, f_name)
        object_list = self.config[data_type]["objects"]
        boxes = []

        with open(label_path, 'r') as f:
            lines = f.readlines() # get rid of \n symbol
            for line in lines:
                bbox = []
                entry = line.split(' ')
                name = entry[0]
                if name in list(object_list.keys()):
                    bbox.append(object_list[name])
                    bbox.extend([float(e) for e in entry[1:]])

                    boxes.append(bbox)


        return np.array(boxes)

    def get_label(self, boxes, geometry):
        '''
        :param boxes: numpy array of shape N:8
        :return: label map: <--- This is the learning target
                a tensor of shape 200 * 175 * 6 representing the expected output
        '''
    
        offset_map = torch.zeros((self.output_shape[0], self.output_shape[1], 2))
        size_map = torch.zeros((self.output_shape[0], self.output_shape[1], 2))
        yaw_map = torch.zeros((self.output_shape[0], self.output_shape[1], 2))
        reg_mask = torch.zeros(self.output_shape)
        
        if self.cls_encoding == "binary":
            cls_map = torch.zeros((self.output_shape[0], self.output_shape[1]), dtype=torch.int64)
        else:
            cls_map = torch.zeros((self.num_classes, self.output_shape[0], self.output_shape[1]))
        
        for i in range(boxes.shape[0]):
            box = boxes[i]
            radius = self.update_cls_map(cls_map, box, geometry)
            self.update_reg_map(offset_map, size_map, yaw_map, reg_mask, radius, box, geometry)

        if self.cls_encoding == "binary":
                cls_map = cls_map.permute(1, 0)
        else:
            cls_map = cls_map.permute(0, 2, 1)
            
        return {
            "cls" : cls_map,
            "offset": offset_map.permute(2, 1, 0),
            "yaw": yaw_map.permute(2, 1, 0),
            "size": size_map.permute(2, 1, 0),
            "reg_mask": reg_mask.permute(1, 0)
        } 


    def get_corners(self, bbox):
        cls, h, w, l, x, y, z, yaw = bbox
        yaw2 = math.fmod(2 * yaw, 2 * math.pi)
        bev_corners = np.zeros((4, 2), dtype=np.float32)


        # rear left
        bev_corners[0, 0] = x - l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
        bev_corners[0, 1] = y - l/2 * np.sin(yaw) + w/2 * np.cos(yaw)

        # rear right
        bev_corners[1, 0] = x - l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
        bev_corners[1, 1] = y - l/2 * np.sin(yaw) - w/2 * np.cos(yaw)

        # front right
        bev_corners[2, 0] = x + l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
        bev_corners[2, 1] = y + l/2 * np.sin(yaw) - w/2 * np.cos(yaw)

        # front left
        bev_corners[3, 0] = x + l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
        bev_corners[3, 1] = y + l/2 * np.sin(yaw) + w/2 * np.cos(yaw)

        reg_target = [np.cos(yaw2), np.sin(yaw2), x, y, w, l]

        return bev_corners, reg_target
        
    def update_cls_map(self, cls_map, box, geometry):
        # box has a form [cls, h, w, l, x, y, z, yaw]
        width = box[2]
        length = box[3]
        width = width / geometry["x_res"] / self.out_size_factor
        length = length / geometry["y_res"] / self.out_size_factor
        radius = 0
        if width > 0 and length > 0:
            radius = gaussian_radius((length, width), min_overlap=self.config['gaussian_overlap'])
            radius = max(self.config['min_radius'], int(radius))
            x, y = box[4], box[5]

            coor_x = (x - geometry["x_min"]) / geometry["x_res"] / self.out_size_factor
            coor_y = (y - geometry["y_min"]) / geometry["y_res"] / self.out_size_factor

            center = torch.tensor([coor_y, coor_x], dtype=torch.float32)
            center_int = center.to(torch.int32)
            
            if self.cls_encoding == "gaussian":
                draw_heatmap_gaussian(cls_map[int(box[0])], center_int, radius)
                radius = radius / 2
            
            elif self.cls_encoding == "inverse_distance":
                x_min = max(0, int(coor_x - radius))
                x_max = min(self.output_shape[0], int(coor_x + radius))

                y_min = max(0, int(coor_y - radius))
                y_max = min(self.output_shape[1], int(coor_y + radius))

                X, Y = np.ogrid[x_min:x_max, y_min:y_max]
                dist_from_center = np.sqrt((X - int(coor_x))**2 + (Y-int(coor_y))**2)
                #print(X.shape)
                #print(x_min, x_max, x, coor_x)
                inv_dist = np.copy(dist_from_center)
                inv_dist[inv_dist == 0] = 1
                inv_dist = 1 / inv_dist

                dist_from_center = torch.from_numpy(dist_from_center)
                inv_dist = torch.from_numpy(inv_dist)

                cls_map[int(box[0])][x_min:x_max, y_min:y_max] = inv_dist
                cls_map[int(box[0])][x_min:x_max, y_min:y_max][dist_from_center > radius] = 0
                cls_map[int(box[0])][x_min:x_max, y_min:y_max][dist_from_center == 1] = 0.8

                radius = radius / 2

            else:
                x_min = max(0, int(coor_x - radius))
                x_max = min(self.output_shape[0], int(coor_x + radius))

                y_min = max(0, int(coor_y - radius))
                y_max = min(self.output_shape[1], int(coor_y + radius))

                X, Y = np.ogrid[x_min:x_max, y_min:y_max]
                dist_from_center = np.sqrt((X - int(coor_x))**2 + (Y-int(coor_y))**2)
                dist_from_center = torch.from_numpy(dist_from_center)
                
                # cls_map[int(box[0]) + 1][x_min:x_max, y_min:y_max][dist_from_center + 0.5 < radius] = 1
                # cls_map[0][x_min:x_max, y_min:y_max][dist_from_center < radius] = 0
                #print(cls_map.shape)
                for p_x in range(x_min, x_max):
                    for p_y in range(y_min, y_max):
                        if math.sqrt((p_x - coor_x)**2 + (p_y - coor_y)**2) < radius:
                            cls_map[p_x][p_y] = int(box[0]) + 1

            return radius
        

    def update_reg_map(self, offset_map, size_map, yaw_map, reg_mask, radius, box, geometry):
        cls, h, w, l, x, y, z, yaw = box
        yaw2 = math.fmod(2 * yaw, 2 * math.pi)

        coor_x = (x - geometry["x_min"]) / geometry["x_res"] / self.out_size_factor
        coor_y = (y - geometry["y_min"]) / geometry["y_res"] / self.out_size_factor

        x_min = max(0, int(coor_x - radius))
        x_max = min(self.output_shape[0], int(coor_x + radius))

        y_min = max(0, int(coor_y - radius))
        y_max = min(self.output_shape[1], int(coor_y + radius))

        for p_x in range(x_min, x_max):
            for p_y in range(y_min, y_max):
                if math.sqrt((p_x - coor_x)**2 + (p_y - coor_y)**2) < radius:
                    metric_x = p_x * self.out_size_factor * geometry["x_res"] + geometry["x_min"]
                    metric_y = p_y * self.out_size_factor * geometry["y_res"] + geometry["y_min"]

                    offset_map[p_x][p_y][0] = x - metric_x
                    offset_map[p_x][p_y][1] = y - metric_y

                    size_map[p_x][p_y][0] = math.log(w)
                    size_map[p_x][p_y][1] = math.log(l)

                    yaw_map[p_x][p_y][0] = math.cos(yaw2)
                    yaw_map[p_x][p_y][1] = math.sin(yaw2)

                    reg_mask[p_x][p_y] = 1

    

    def read_bbox(self, boxes):
        corner_list = []
        class_list = []
        for i in range(boxes.shape[0]):
            box = boxes[i]
            class_list.append(box[0])
            corners = self.get3D_corners(box)
            corner_list.append(corners)
        return (class_list, corner_list)

    def filter_boxes(self, boxes, data_type):
        filtered_boxes = []
        geometry = self.config[data_type]["geometry"]
        for i in range(boxes.shape[0]):
            box = boxes[i]
            x, y = box[4:6]
            
            if (x > geometry["x_min"]) and (x < geometry["x_max"]) and (y > geometry["y_min"]) and (y < geometry["y_max"]):
                filtered_boxes.append(box)
            
        return torch.from_numpy(np.array(filtered_boxes))


    def get3D_corners(self, bbox):
        h, w, l, x, y, z, yaw = bbox[1:]

        corners = []
        front = l / 2
        back = -l / 2
        left = w / 2
        right = -w / 2
        top = h
        bottom = 0
        corners.append([front, left, top])
        corners.append([front, left, bottom])
        corners.append([front, right, bottom])
        corners.append([front, right, top])
        corners.append([back, left, top])
        corners.append([back, left, bottom])
        corners.append([back, right, bottom])
        corners.append([back, right, top])
        
        for i in range(8):
            corners[i] = self.rotate_pointZ(corners[i], yaw)[0] + np.array([x, y, z])
       
        return corners


    def rotate_pointZ(self, point, yaw):
        rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                    [np.sin(yaw), np.cos(yaw), 0],
                                    [0, 0, 1]])

        rotated_point = np.matmul(rotation_matrix, np.reshape(point, (3, 1)))
        return np.reshape(rotated_point, (1, 3))

    def get_transforms(self, config):
        transforms = []
        if config["rotation"]["use"]:
            limit_angle = config["rotation"]["limit_angle"]
            p = config["rotation"]["p"]
            transforms.append(Random_Rotation(limit_angle, p))
        if config["scaling"]["use"]:
            range = config["scaling"]["range"]
            p = config["scaling"]["p"]
            transforms.append(Random_Scaling(range, p))

        if config["translation"]["use"]:
            scale = config["translation"]["scale"]
            p = config["translation"]["p"]
            transforms.append(Random_Translation(scale, p))

        return transforms

    def create_data_list(self):
        data_list = []
        data_type_list = []
        with open(self.data_file, "r") as f:
            for line in f:
                line = line.strip()
                data, data_type = line.split(";")
                data_list.append(data)
                data_type_list.append(data_type)
        
        self.data_list = data_list
        self.data_type_list = data_type_list


if __name__ == "__main__":
    data_file = "/Users/hoangtran/miniforge3/envs/lidar-perception-master/data-3d/training_data_JRDB/train_text/train.txt"

    with open("/Users/hoangtran/miniforge3/envs/lidar-perception-master/data-3d/training_data_JRDB/labels_3d/bytes-cafe-2019-02-07_0.json", 'r') as f:
        config = json.load(f)

    dataset = Dataset(data_file, config["data"], config["augmentation"])

    # for data in dataset:
    #     label = data["label"]
    #     voxel = data["voxel"]
    #     #voxel = voxel.permute(1, 2, 0)
    #     #print(voxel.shape)
    #     #print(torch.sum(voxel, axis = 2))
    #     #print(label[:, 0].shape)
    #     #imgplot = plt.imshow(label[:, :, 0])
    #     #plt.imshow(torch.sum(voxel, axis = 2), cmap="brg", vmin=0, vmax=255)
    #     #img = voxel_to_img(voxel)
    #     #imgplot = plt.imshow(img)
    #    # plt.show()
    #     break