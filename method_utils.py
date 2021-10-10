import os
import sys
import csv
from collections import OrderedDict

import numpy as np
import torch.utils.data
import cv2
import matplotlib.pyplot as plt


def make_datapath_list(root_path):
    """
    動画を画像データにしたフォルダへのファイルパスリストを作成する。    root_path : str、データフォルダへのrootパス
    Returns：ret : video_list、動画を画像データにしたフォルダ直上のファイルパスリスト
    """

    # 動画を画像データにしたフォルダへのファイルパスリスト
    video_list = list()

    # root_pathにある、クラスの種類とパスを取得
    class_list = os.listdir(path=root_path)

    # 各クラスの動画ファイルを画像化したフォルダへのパスを取得
    for class_list_i in (class_list):  # クラスごとのループ

        # クラスのフォルダへのパスを取得
        class_path = os.path.join(root_path, class_list_i)

        # 各クラスのフォルダ内の画像フォルダを取得するループ
        for file_name in os.listdir(class_path):
            # print(file_name)

            # ファイル名と拡張子に分割
            name, ext = os.path.splitext(file_name)

            # フォルダでないmp4ファイルは無視
            if ext == '.mp4':
                continue

            # 動画ファイルを画像に分割して保存したフォルダのパスを取得
            video_img_directory_path = os.path.join(class_path, name)

            # vieo_listに追加
            video_list.append(video_img_directory_path)

    return video_list


def get_c2i_i2c_from_dir_hrc(root_path: str, sort_order=[]):
    """
    Loading class name and ID from directory structure.
    (e.g.)
    root_path
    |-- classA
        |--  movie1
        |--  movie2
        :
    |-- classB
    :


    The folder names under "root_path" must be set as class name,
    and all ID are set automatically unless sorted by the setting of list "sort_order"

    Elements of "sort order" must be correspondent to class names.

    @param root_path:
    @param sort_order:
    @return:
    """
    class_list = os.listdir(path=root_path)
    if sort_order != []:
        if set(sort_order) <= set(sort_order):
            class_list = sort_order
            pass
        else:
            print('Set Correct class name')
            sys.exit()
        pass
    else:
        pass
    cls2id = OrderedDict()
    id2cls = OrderedDict()
    for idx, cls_name in enumerate(class_list):
        id2cls[idx] = cls_name
        cls2id[cls_name] = idx
        pass
    return cls2id, id2cls


def get_label_id_dictionary(label_dicitionary_path):
    label_id_dict = {}
    id_label_dict = {}

    # eoncodingはUbuntuもこれで良いのか、確認せねば
    with open(label_dicitionary_path, encoding="utf-8_sig") as f:
        # 読み込む
        reader = csv.DictReader(f, delimiter=",", quotechar='"')

        # 1行ずつ読み込み、辞書型変数に追加します
        for row in reader:
            label_id_dict.setdefault(
                row["class_label"], int(row["label_id"]) - 1)
            id_label_dict.setdefault(
                int(row["label_id"]) - 1, row["class_label"])

    return label_id_dict, id_label_dict


def get_mean_and_std(dataloader: torch.utils.data.DataLoader):
    # Get average image
    nimages = 0
    mean = 0.
    std = 0.

    for batch in dataloader:
        tensor_ = batch[0]  # [B, L, C, W, H]
        # Rearrange batch to be the shape of [B*L, C, W*H]
        tensor_ = tensor_.view(tensor_.size(0) * tensor_.size(1), tensor_.size(2), -1)
        # Update total number of images
        nimages += tensor_.size(0)
        # Compute mean and std here
        mean += tensor_.mean(2).sum(0)
        std += tensor_.std(2).sum(0)
        pass
    # Final step
    mean /= nimages
    std /= nimages
    print("mean: ", mean)
    print("std: ", std)


def show_img(tensors: torch.tensor, batch_id: int, seq_id: int):
    """
    Loading the tensor the shape of which is [B, L, H, W, CH], and show by matplotlib function.
    That image is specified by B=batch_id, L=seq_id

    @param tensors:
    @param batch_id:
    @param seq_id:
    @return:
    """
    ch_h_w = tensors[batch_id][seq_id].to('cpu').detach().numpy().astype(np.uint8).copy()
    img = ch_h_w.transpose(1, 2, 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    return img
