import os
import sys
import csv
from collections import OrderedDict
import glob

import numpy as np
import torch.utils.data
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm


class MultiSegmentVideoList:
    def __init__(self, root_path,
                 seg_span,
                 seg_len,
                 strt_index=1,
                 search_ext="*",
                 ignore_exts=[".mp4", ".txt", ".csv", ".zip"],
                 shift_inflation=False,
                 inflation_ratio=1.,
                 ):
        """

        @param root_path:
        @param seg_span:
        @param seg_len:
        @param strt_index:
        @param search_ext:
        @param ignore_exts:
        @param shift_inflation:
        @param inflation_ratio:
        """
        assert strt_index >= 1, "Set variable strt_index above 1, (default: 1)"
        self.shift_inflation = shift_inflation
        self.inflation_ratio = inflation_ratio  # TODO implementation
        # Data creation
        self.video_list = []
        self.top_images = []
        self.total_frames = []
        # Top indices
        self.video_list2top_indices = {}
        self.video_list2top_files = {}
        # End indices
        self.video_list2end_indices = {}
        self.video_list2end_files = {}
        # Indices matrix
        self.video_list2mat = {}
        self.video_list2mat_rect = {}

        # top_path2indices
        self.top_file2indices = {}

        self._make_datapath_with_mutiseg(root_path=root_path,
                                         seg_span=seg_span,
                                         seg_len=seg_len,
                                         strt_index=strt_index,
                                         search_ext=search_ext,
                                         ignore_exts=ignore_exts
                                         )

    def __len__(self):
        return self.__total_segments

    def __call__(self, *args, **kwargs):
        return self.top_images, self.top_file2indices

    def _make_datapath_with_mutiseg(self, root_path, seg_span, seg_len, strt_index, search_ext, ignore_exts):

        total_segments = 0
        class_list = os.listdir(path=root_path)
        for class_list_i in (class_list):  # クラスごとのループ
            # クラスのフォルダへのパスを取得
            class_path = os.path.join(root_path, class_list_i)
            for file_name in os.listdir(class_path):
                img_stored_dir = os.path.join(class_path, file_name)
                image_files = sorted(glob.glob(os.path.join(img_stored_dir, search_ext)))
                imag_num = len(image_files)
                max_index = imag_num - 1
                # Count the number of available segment
                # Because top_indices will be used for file search index from the result of glob.glob()
                # image_0001.jpeg is equal to ID=0
                top_indices = np.arange(start=strt_index - 1, stop=max_index, step=seg_span * (seg_len - 1) + 1)
                # top_indices = np.arange(start=strt_index, stop=max_index, step=seg_span * (seg_len - 1) + 1)
                if self.shift_inflation:
                    top_indices = np.arange(start=strt_index, stop=max_index)
                    pass
                end_indices = top_indices + seg_span * (seg_len - 1)
                rectified_end_indices = np.where((end_indices > max_index), -1, end_indices)
                # Because indices_mat will be used for file search
                # by identifying whether that element is included in file name or NOT,
                # image_0001.jpeg is equal to ID=1
                indices_mat = np.array(
                    [np.arange(start=1 + top_indices[i], stop=end_indices[i] + seg_span, step=seg_span) for i in
                     range(len(top_indices))])
                rectified_indices_mat = np.where((indices_mat > max_index), -1, indices_mat)
                # print(file_name)
                # print(top_indices)
                # print(end_indices)
                # print(rectified_end_indices)
                # print(rectified_indices_mat)
                # print(len(top_indices))
                top_files = [image_files[elem] for elem in top_indices]
                end_files = [image_files[elem] for elem in rectified_end_indices]

                # Remove NOT target path by specified extension
                name, ext = os.path.splitext(file_name)
                if ext in ignore_exts:
                    continue

                # Set variables
                video_img_directory_path = os.path.join(class_path, name)
                self.video_list.append(video_img_directory_path)
                self.top_images.extend(top_files)
                self.total_frames.append(imag_num)
                self.video_list2top_indices[video_img_directory_path] = top_indices
                self.video_list2top_files[video_img_directory_path] = top_files
                self.video_list2end_indices[video_img_directory_path] = end_indices
                self.video_list2end_files[video_img_directory_path] = end_files
                self.video_list2mat[video_img_directory_path] = indices_mat
                self.video_list2mat_rect[video_img_directory_path] = rectified_indices_mat
                total_segments += len(top_indices)

                # for VideoDataset
                for i, top_filename in enumerate(top_files):
                    self.top_file2indices[top_filename] = rectified_indices_mat[i]
                    pass

                pass
            pass
        self.__total_segments = total_segments
        pass


def make_anno_list(root_path):
    """
    Loading paths to annotations files
    The structure must be similar to the video dataset

    @param root_path: the path to annotations stored
    @return: the list of them
    """
    anno_paths = []
    class_list = os.listdir(path=root_path)
    for class_list_i in (class_list):
        # クラスのフォルダへのパスを取得
        class_path = os.path.join(root_path, class_list_i)
        # print(class_path)
        # 各クラスのフォルダ内の画像フォルダを取得するループ
        for file_name in sorted(glob.glob(os.path.join(class_path, "*"))):
            anno_paths.append(file_name)
            pass

    return anno_paths


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
    print('Analyzing all batch ...')
    with tqdm(total=len(dataloader)) as pbar:
        for batch in dataloader:
            tensor_ = batch[0]  # [B, L, C, W, H]
            # Rearrange batch to be the shape of [B*L, C, W*H]
            tensor_ = tensor_.view(tensor_.size(0) * tensor_.size(1), tensor_.size(2), -1)
            # Update total number of images
            nimages += tensor_.size(0)
            # Compute mean and std here
            mean += tensor_.mean(2).sum(0)
            std += tensor_.std(2).sum(0)
            pbar.update(1)
            pass
    # Final step
    mean /= nimages
    std /= nimages
    print("mean: ", mean)
    print("std: ", std)


def _tensor2cvmat_rgb(tensors: torch.tensor, batch_id: int, seq_id: int):
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
    return img


def show_img(tensors: torch.tensor, batch_id: int, seq_id: int):
    """
    Show laod image as plt.plot()

    @param tensors:
    @param batch_id:
    @param seq_id:
    @return:
    """
    img = _tensor2cvmat_rgb(tensors, batch_id, seq_id)
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    return img


def save_input_animation(tensors: torch.tensor, batch_id: int,
                         interval_ms=300, repeat_delay_ms=1000, ani_name="test.gif"):
    fig = plt.figure()
    ims = [[plt.imshow(_tensor2cvmat_rgb(tensors, batch_id, i))] for i in
           range(tensors.shape[1])]  # require double-list shape
    ani = animation.ArtistAnimation(fig, ims, interval=interval_ms, repeat_delay=repeat_delay_ms)
    ani.save(ani_name)
    pass
