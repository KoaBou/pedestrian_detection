import numpy as np
import math


def center_to_corner_box3d(boxes_center):
    # (N, 7) -> (N, 8, 3)
    N = boxes_center.shape[0]
    ret = np.zeros((N, 8, 3), dtype=np.float32)

    for i in range(N):
        box = boxes_center[i]
        translation = box[3:6]
        size = box[0:3]
        rotation = [0, 0, box[-1]]

        h, w, l = size[0], size[1], size[2]
        trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
            [0, 0, 0, 0, h, h, h, h]])
        
 
        # re-create 3D bounding box in velodyne coordinate system
        yaw = rotation[2]
        rotMat = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0]])
        cornerPosInVelo = np.dot(rotMat, trackletBox) + \
                          np.tile(translation, (8, 1)).T
        box3d = cornerPosInVelo.transpose()
        ret[i] = box3d

    return ret

def corner_to_center_box3d(boxes_corner):
    # (N, 8, 3) -> (N, 7) 
    ret = []
    for roi in boxes_corner:
        roi = np.array(roi)
        h = abs(np.sum(roi[:4, 2] - roi[4:, 2]) / 4)

        l = np.sum(
            np.sqrt(np.sum((roi[0, [0, 1]] - roi[3, [0, 1]]) ** 2)) +
            np.sqrt(np.sum((roi[1, [0, 1]] - roi[2, [0, 1]]) ** 2)) +
            np.sqrt(np.sum((roi[4, [0, 1]] - roi[7, [0, 1]]) ** 2)) +
            np.sqrt(np.sum((roi[5, [0, 1]] - roi[6, [0, 1]]) ** 2))
        ) / 4
        w = np.sum(
            np.sqrt(np.sum((roi[0, [0, 1]] - roi[1, [0, 1]]) ** 2)) +
            np.sqrt(np.sum((roi[2, [0, 1]] - roi[3, [0, 1]]) ** 2)) +
            np.sqrt(np.sum((roi[4, [0, 1]] - roi[5, [0, 1]]) ** 2)) +
            np.sqrt(np.sum((roi[6, [0, 1]] - roi[7, [0, 1]]) ** 2))
        ) / 4
        x = np.sum(roi[:, 0], axis=0) / 8
        y = np.sum(roi[:, 1], axis=0) / 8
        z = np.sum(roi[0:4, 2], axis=0) / 4
        rz = np.sum(
            math.atan2(roi[2, 0] - roi[1, 0], -roi[2, 1] + roi[1, 1]) +
            math.atan2(roi[6, 0] - roi[5, 0], -roi[6, 1] + roi[5, 1]) +
            math.atan2(roi[3, 0] - roi[0, 0], -roi[3, 1] + roi[0, 1]) +
            math.atan2(roi[7, 0] - roi[4, 0], -roi[7, 1] + roi[4, 1]) 

        ) / 4
       
        rz = rz - np.pi / 2
        ret.append([h, w, l, x, y, z, rz])

    return np.array(ret)

def corner_to_center_box2d(boxes_corner):
    # (N, 4, 2) -> (N, 5)
    ret = []
    for roi in boxes_corner:
        roi = np.array(roi)

        l = np.sum(
            np.sqrt(np.sum((roi[0, [0, 1]] - roi[3, [0, 1]]) ** 2)) +
            np.sqrt(np.sum((roi[1, [0, 1]] - roi[2, [0, 1]]) ** 2)) 
        ) / 2
        w = np.sum(
            np.sqrt(np.sum((roi[0, [0, 1]] - roi[1, [0, 1]]) ** 2)) +
            np.sqrt(np.sum((roi[2, [0, 1]] - roi[3, [0, 1]]) ** 2)) 
        ) / 2
        x = np.sum(roi[:, 0], axis=0) / 4
        y = np.sum(roi[:, 1], axis=0) / 4

        rz = np.sum(
            math.atan2(roi[2, 0] - roi[1, 0], -roi[2, 1] + roi[1, 1]) +
            math.atan2(roi[3, 0] - roi[0, 0], -roi[3, 1] + roi[0, 1])
        ) / 2
       
        rz = rz - np.pi / 2
        ret.append([x, y, l, w, rz])

    return np.array(ret)


def point_transform(points, tx, ty, tz, rx=0, ry=0, rz=0):
    # Input:
    #   points: (N, 3)
    #   rx/y/z: in radians
    # Output:
    #   points: (N, 3)
    N = points.shape[0]
    points = np.hstack([points, np.ones((N, 1))])

    mat1 = np.eye(4)
    mat1[3, 0:3] = tx, ty, tz
    points = np.matmul(points, mat1)

    if rx != 0:
        mat = np.zeros((4, 4))
        mat[0, 0] = 1
        mat[3, 3] = 1
        mat[1, 1] = np.cos(rx)
        mat[1, 2] = -np.sin(rx)
        mat[2, 1] = np.sin(rx)
        mat[2, 2] = np.cos(rx)
        points = np.matmul(points, mat)

    if ry != 0:
        mat = np.zeros((4, 4))
        mat[1, 1] = 1
        mat[3, 3] = 1
        mat[0, 0] = np.cos(ry)
        mat[0, 2] = np.sin(ry)
        mat[2, 0] = -np.sin(ry)
        mat[2, 2] = np.cos(ry)
        points = np.matmul(points, mat)

    if rz != 0:
        mat = np.zeros((4, 4))
        mat[2, 2] = 1
        mat[3, 3] = 1
        mat[0, 0] = np.cos(rz)
        mat[0, 1] = -np.sin(rz)
        mat[1, 0] = np.sin(rz)
        mat[1, 1] = np.cos(rz)
        points = np.matmul(points, mat)

    return points[:, 0:3]


def box_transform(boxes, tx, ty, tz, r=0):
    # Input:
    #   boxes: (N, 7) h w l x y z rz/y
    # Output:
    #   boxes: (N, 7) h w l x y z rz/y
    boxes_corner = center_to_corner_box3d(boxes)  # (N, 8, 3)
    for idx in range(len(boxes_corner)):
        boxes_corner[idx] = point_transform(
            boxes_corner[idx], tx, ty, tz, rz=r)


    return corner_to_center_box3d(boxes_corner)


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


class Compose(object):
    def __init__(self, transforms, p=1.0):
        self.transforms = transforms
        self.p = p

    def __call__(self, lidar, labels):
        if np.random.random() <= self.p:
            for t in self.transforms:
                lidar, labels = t(lidar, labels)
        return lidar, labels


class OneOf(object):
    def __init__(self, transforms, p=1.0):
        self.transforms = transforms
        self.p = p

    def __call__(self, lidar, labels):
        if np.random.random() <= self.p and len(self.transforms) > 0:
            choice = np.random.randint(low=0, high=len(self.transforms))
            lidar, labels = self.transforms[choice](lidar, labels)

        return lidar, labels


class Random_Rotation(object):
    def __init__(self, limit_angle=20., p=0.5):
        self.limit_angle = limit_angle / 180. * np.pi
        self.p = p

    def __call__(self, lidar, labels):
        """
        :param labels: # (N', 7) h, w, l, x, y, z,  r
        :return:
        """
        if np.random.random() <= self.p:
            angle = np.random.uniform(-self.limit_angle, self.limit_angle)
            lidar[:, 0:3] = point_transform(lidar[:, 0:3], 0, 0, 0, rz=angle)
            labels = box_transform(labels, 0, 0, 0, r=angle)

        return lidar, labels


class Random_Scaling(object):
    def __init__(self, scaling_range=(0.95, 1.05), p=0.5):
        self.scaling_range = scaling_range
        self.p = p

    def __call__(self, lidar, labels):
        """
        :param labels: # (N', 7) h, w, l, x, y, z,  r
        :return:
        """
        if np.random.random() <= self.p:
            factor = np.random.uniform(self.scaling_range[0], self.scaling_range[0])
            lidar[:, 0:3] = lidar[:, 0:3] * factor
            labels[:, 0:6] = labels[:, 0:6] * factor

        return lidar, labels


class Random_Translation(object):
    def __init__(self, scale = 0.1, p = 0.5):
        self.scale = scale
        self.p = p

    def __call__(self, lidar, labels):
        """
        :param labels: # (N', 7) h, w, l, x, y, z,  r
        :return:
        """
        if np.random.random() <= self.p:
            dx = np.random.normal(loc = 0.0, scale = self.scale)
            dy = np.random.normal(loc = 0.0, scale = self.scale)
            dz = np.random.normal(loc = 0.0, scale = self.scale)

            lidar[:, 0] += dx
            lidar[:, 1] += dy
            lidar[:, 2] += dz
            labels[:, 3] +=dx
            labels[:, 4] +=dy
            labels[:, 5] +=dz

        return lidar, labels