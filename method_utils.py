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
    def __init__(self, root_path,  # TODO rename `root_path` to appropriate name
                 seg_span,
                 seg_len,
                 strt_index=1,
                 search_ext="*",
                 ignore_exts=[".mp4", ".txt", ".csv", ".zip"],
                 shift_inflation=False,
                 inflation_ratio=1.,
                 load_from="root_path",
                 ):
        """
        Give root_path, Search the directory structure and get video files to read and corresponding indices

        @param root_path: Directory name to search dataset structure such as `data/kinetics_images`.
                            What type of `root_path` was set is specified by `load_from` option
        @param seg_span:
        @param seg_len:
        @param strt_index:
        @param search_ext:
        @param ignore_exts:
        @param shift_inflation:
        @param inflation_ratio:
        @param load_from: Settable argument are defined in following __ACCEPTABLE_LOAD_FROM_STYLE
                            root_path ... a top directory for dataset (e.g. `root_path` in README.md)
                            image_stored_dir ... a top directory for image files
                                                (e.g. (e.g. `root_path/classA/hoge` in README.md))
                            movie_file ... #TODO NOT implemented
        """
        __ACCEPTABLE_LOAD_FROM_STYLE = ["root_path", "image_stored_dir", "movie_file"]
        assert strt_index >= 1, "Set variable strt_index above 1, (default: 1)"  # TODO Correct to read image_00000.jpg
        assert load_from in __ACCEPTABLE_LOAD_FROM_STYLE, "Set argument `load_from` from {} (Default : {})".format(
            __ACCEPTABLE_LOAD_FROM_STYLE, __ACCEPTABLE_LOAD_FROM_STYLE[0])
        self.shift_inflation = shift_inflation
        self.inflation_ratio = inflation_ratio  # TODO implemented
        # Data creation
        self.total_frames = []  # TODO make the return of __call__()
        self.total_segments = 0

        if load_from == "root_path":
            self.top_images, self.top_file2indices, self.total_frames, self.total_segments = \
                self._make_datapath_with_mutiseg(
                    root_path=root_path,
                    seg_span=seg_span,
                    seg_len=seg_len,
                    strt_index=strt_index,
                    search_ext=search_ext,
                    ignore_exts=ignore_exts
                )
        elif load_from == "image_stored_dir":
            top_images, rectified_indices_mat, segments_num = \
                self._create_indices_by_filename(
                    img_stored_dir=root_path,
                    seg_span=seg_span,
                    seg_len=seg_len,
                    strt_index=strt_index,
                    search_ext=search_ext,
                )
            self.top_images = top_images
            self.top_file2indices = {top_image: rectified_indices_mat[i] for i, top_image in enumerate(top_images)}
            self.total_frames = [segments_num]
            self.total_segments = segments_num
            pass
        else:
            raise NotImplementedError()

    def __len__(self):
        return self.total_segments

    def __call__(self, *args, **kwargs):
        """


        @param args:
        @param kwargs:
        @return:
            self.top_images : Extended list of `top_files` given by _create_indices_by_filename()
            self.top_file2indices : Extended matrix of `rectified_indices_mat` given by _create_indices_by_filename()
        """
        return self.top_images, self.top_file2indices

    def _create_indices_by_filename(self, img_stored_dir: str, seg_span, seg_len, strt_index, search_ext):
        """
        Searching all files (having extension `search_ext`) in img_stored_dir and by seg_span, seg_len and strt_index.

        @param img_stored_dir:
        @param seg_span:
        @param seg_len:
        @param strt_index:
        @param search_ext:
        @return:
            top_files: List of filenames like [<path-to>/image_0001.jpeg, <path-to>/image_0030.jpeg, ..., ]
                        which is recognized as the first step of time series input
            rectified_indices_mat: Matrix shape of which is (len(top_files), seg_len) and composed of integer,
                        reveal the indices to loading file (e.g. as default, 1 means loading image_0001.jpeg)
            segments_num: The total image numbers saved in `img_stored_dir`
        """
        image_files = sorted(glob.glob(os.path.join(img_stored_dir, search_ext)))

        imag_num = len(image_files)
        max_index = imag_num
        # Count the number of available segment
        # Because top_indices will be used for file search index from the result of glob.glob()
        # image_0001.jpeg is equal to ID=0
        top_indices = np.arange(start=strt_index - 1, stop=max_index, step=seg_span * (seg_len - 1) + 1)
        if self.shift_inflation:
            top_indices = np.arange(start=strt_index - 1, stop=max_index)
            pass
        end_indices = top_indices + seg_span * (seg_len - 1)
        # Because indices_mat will be used for file search
        # by identifying whether that element is included in file name or NOT,
        # image_0001.jpeg is equal to ID=1
        indices_mat = np.array(
            [np.arange(start=1 + top_indices[i], stop=end_indices[i] + seg_span + 1, step=seg_span) for i in
             range(len(top_indices))])
        rectified_indices_mat = np.where((indices_mat > max_index), -1, indices_mat)
        top_files = [image_files[elem] for elem in top_indices]
        segments_num = len(top_files)
        return top_files, rectified_indices_mat, segments_num

    def _make_datapath_with_mutiseg(self, root_path, seg_span, seg_len, strt_index, search_ext, ignore_exts):
        # Initialize returns
        top_images = []
        top_file2indices = {}
        total_frames = []

        # Load from root_path
        class_list = os.listdir(path=root_path)
        for class_list_i in (class_list):  # クラスごとのループ
            # クラスのフォルダへのパスを取得
            class_path = os.path.join(root_path, class_list_i)
            for file_name in os.listdir(class_path):
                img_stored_dir = os.path.join(class_path, file_name)
                top_files, rectified_indices_mat, segments_num = self._create_indices_by_filename(img_stored_dir,
                                                                                                  seg_span,
                                                                                                  seg_len,
                                                                                                  strt_index,
                                                                                                  search_ext)
                # Remove NOT target path by specified extension
                # TODO: Reveal whether it is necessary
                name, ext = os.path.splitext(file_name)
                if ext in ignore_exts:
                    continue

                # Set variables
                top_images.extend(top_files)
                total_frames.append(segments_num)

                # for VideoDataset
                for i, top_filename in enumerate(top_files):
                    top_file2indices[top_filename] = rectified_indices_mat[i]
                    pass
                pass
            pass
        total_frame_sum = np.sum(np.array(total_frames))
        # print(top_file2indices)
        return top_images, top_file2indices, total_frames, total_frame_sum


def get_top_imagefilename(dir_path_to_search: str, image_ext=".jpg", strt_index=1) -> str:
    """
    Get top image filename to feed the class MultiSegmentVideoList()

    @param dir_path_to_search:
    @param image_ext:
    @param strt_index:
    @return:
    """
    return sorted(glob.glob(os.path.join(dir_path_to_search, "*" + image_ext)))[strt_index - 1]


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
