import os
import random

TRAIN_NAME = "human_train.txt"
VAL_NAME = "human_val.txt"

def split(datas, percents, save_folder):
    train_set = []
    val_set = []
    for data in datas.keys():
        lidar_files = sorted(os.listdir(datas[data]))
        random.shuffle(lidar_files)
        train_len = int(percents[data] * len(lidar_files) / 100)
        val_len = train_len // 8

        train_set = lidar_files[:train_len]
        val_set = lidar_files[train_len: min(train_len + val_len, len(lidar_files))]

        write_to_file(os.path.join(save_folder, TRAIN_NAME), train_set, data)
        write_to_file(os.path.join(save_folder, VAL_NAME), val_set, data)


def write_to_file(filename, dataset, datatype):
    with open(filename, 'a') as f:
        for data in dataset:
            f.write(data.split(".")[0] + ";" + datatype)
            f.write('\n')


if __name__ == "__main__":

    datas = {"jrdb": "/home/thuong/data-3d/jrdb/pointcloud",
             }
    percents = {"jrdb": 90}

    save_folder = "/home/thuong/data-3d/list"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    split(datas, percents, save_folder)
