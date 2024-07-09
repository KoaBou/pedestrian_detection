import torch
import json
from models.torchscript.torchscript_model import TorchscriptModel

if __name__ == "__main__":
    geometry = {
                "y_min": -40.0,
                "y_max": 40.0,
                "x_min": -35.0,
                "x_max": 35.0,
                "z_min": -2.5,
                "z_max": 1.0,
                "x_res": 0.1,
                "y_res": 0.1,
                "z_res": 0.1
            }

    with open("/home/thuong/lidar-objectdetection/detection_stpc/configs/mobilepixor-bigobjs.json", 'r') as f:
        config = json.load(f)
    model_path = "/home/thuong/experiments/mobilepixor_gaussian_synthesis-shuttle-bigobjs_21-10-2022_1/checkpoints/198epoch"
    data_file = "/home/thuong/clean_data/list/synthesis_shuttle_val.txt"

    model = TorchscriptModel(config["model"], config["data"]["out_size_factor"], config["data"]["num_classes"])
    #model.to(config['device'])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    #device = config["device"]
    model.half()
    scripted_model = torch.jit.script(model)

    print(scripted_model.code)
    scripted_model.save("/home/thuong/models/mobilepixor_gauss_big_objs_221101.pt")