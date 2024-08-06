#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import time
import torch
from torch import nn
import json
from sensor_msgs_py import point_cloud2 as pc2
from utils.preprocess import voxelize
from utils.postprocess import filter_pred

from sensor_msgs.msg import PointCloud2
from vision_msgs.msg import BoundingBox3D, BoundingBox3DArray, Detection3D,  Detection3DArray, ObjectHypothesisWithPose
from IRBGHR_PIXOR.models.model import CustomModel

class Eval():
  def __init__(self, data_loader, model, device, img_size, config):
    self.data_loader = data_loader
    self.model = model
    self.model.eval()
    self.device = device
    self.image_size = img_size
    self.config = config

class LiDARPedestrianDetection(Node):
    def __init__(self):
        super().__init__("lidar_detection_node")
        self.subscription = self.create_subscription(
            PointCloud2,
            "/jrdb/point_cloud/lower",
            self.lidar_cb,
            10)
        
        self.pred_pub = self.create_publisher(
            Detection3DArray,
            "/preds",
            10)
        
        config_path = "/home/ngin/pcl_detect_ws/src/lidar_detection/scripts/base_demo.json"
        with open(config_path, 'r') as f:
            self.config = json.load(f) 
        self.model = CustomModel(self.config["model"], self.config["data"]["num_classes"])  
        # Replace MyModel with your actual model class
        model_path = '/home/ngin/pcl_detect_ws/src/lidar_detection/18epoch'
        if torch.cuda.is_available():
            print("cuda")
            self.device = torch.device("cuda")
        else: 
            print("cpu")
            self.device = torch.device("cpu")
        model_state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)
        self.model.eval()
  
    # Haven't test this function yet 
    # def yaw_to_quaternion(self, yaw):
    #     # Helper function to convert a yaw angle (in radians) to a quaternion
    #     import math
    #     half_yaw = yaw * 0.5
    #     cy = math.cos(half_yaw)
    #     sy = math.sin(half_yaw)
    #     return (0, 0, sy, cy)

    def extract_bboxes(self, predictions, header):
        config = self.config['data']['jrdb']
        out_size_factor = self.config['data']['out_size_factor']
        thres = 0.3  # Threshold for filtering predictions
        nms_thres = None  # NMS threshold
        
        boxes = filter_pred(predictions, config, out_size_factor, thres, nms_thres)
        bbox_array = BoundingBox3DArray()
        bbox_array.header = header


        pred_array = Detection3DArray()
        pred_array.header = header
        pred_array.header.frame_id = 'base_link'


        for idx, box in enumerate(boxes):
            bbox = BoundingBox3D()
            bbox.center.position.x, bbox.center.position.y, \
            bbox.center.position.z = box[2], box[3], 0.0  # Assuming z=0 for simplicity
            print(bbox.center.position.x, bbox.center.position.y, bbox.center.position.z)
            bbox.size.x, bbox.size.y, bbox.size.z = box[5], box[4], 2.0  # Fixed height for simplicity
            # Convert yaw to quaternion (haven't test yet)
            # quaternion = self.yaw_to_quaternion(box[6])
            # bbox.orientation.x, bbox.orientation.y, bbox.orientation.z, bbox.orientation.w = quaternion
            bbox_array.boxes.append(bbox)

            pred = Detection3D()
            pred.bbox = bbox
            pred.header.frame_id = 'base_link'
            
            result = ObjectHypothesisWithPose()
            result.id = str(idx)
            result.score = box[1]
            pred.results.append(result)

            pred_array.detections.append(pred)

        return pred_array

    def lidar_cb(self, msg):
        # Input
        start_time = time.time()
        geometry = self.config['data']['jrdb']['geometry']
        raw_points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
        # Convert to bev 
        voxel = voxelize(raw_points, geometry) 
        voxel_tensor = torch.tensor(voxel).float().unsqueeze(0).to(self.device)
        voxel_tensor = voxel_tensor.permute(0, 3, 1, 2)
        # Predict and extract to bbox
        with torch.no_grad():
            predictions = self.model(voxel_tensor)
        pred_array = self.extract_bboxes(predictions, msg.header)
        self.pred_pub.publish(pred_array)
        print(f"Processing time: {time.time() - start_time}s, Length: {len(voxel)}")
        
def main(args=None):
    rclpy.init(args=args)
    detector = LiDARPedestrianDetection()
    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
