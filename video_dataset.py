# URL: https://github.com/KyanainsGate/pytorch_advanced/blob/master/9_video_classification_eco/utils/kinetics400_eco_dataloader.py

# 第9章 動画分類（ECO：Efficient 3DCNN）
# 9.4	Kinetics動画データセットからDataLoaderの作成

# 必要なパッケージのimport
import os
from multiprocessing import Pool
import time

from PIL import Image
import numpy as np
import torch.utils.data


def _load_psuedo_true_img(filename):
    if "[pad]" in filename:
        size_info = filename.split("[pad]")[-1]
        h_ = int(size_info.split("x")[0])
        w_ = int(size_info.split("x")[1])
        ch = int(size_info.split("x")[2])
        img = Image.fromarray(np.zeros((h_, w_, ch), np.uint8))
    else:
        img = Image.open(filename).convert('RGB')
    return img


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_list, label_id_dict, seg_span, seg_num,
                 phase, transform, slash_sp_num=1, path_delim="\\",
                 img_tmpl='image_{:05d}.jpg',
                 strt_index=1,
                 cpu_thread=1,
                 ):
        """
        Constructor for building torch.Dataset

        @param video_list:
        @param label_id_dict:
        @param seg_span:
        @param seg_num:
        @param phase:
        @param transform:
        @param slash_sp_num:
        @param path_delim:
        @param img_tmpl:
        @param strt_index:
        @param cpu_thread:
        """
        self.video_list = video_list  # 動画画像のフォルダへのパスリスト
        self.label_id_dict = label_id_dict  # ラベル名をidに変換する辞書型変数
        self.seg_span = seg_span  # Define delta t of images
        self.seg_num = seg_num  # Fixed time series length to feed,
        self.phase = phase  # train or val
        self.transform = transform  # 前処理
        self.slash_sp_num = slash_sp_num
        self.img_tmpl = img_tmpl  # 読み込みたい画像のファイル名のテンプレート
        self.strt_index = strt_index
        self.debug_cnt = 1
        self.path_delim = path_delim  # "/" for Linux, "\\" for windows
        self.cpu_thread = cpu_thread  # Set thread num for loading image parallelly
        self.null_img_num = 0

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        """
        Get index ID when called by inter() and return all input's correspondence

        :param index:
        :return: transformed images ... (torch.torch.(5D tensor of [b, l, h, w, ch])),
                 label (list) ... the strings of labels
                 label_id (torch.torch(1D class ID)),
                 dir_path (list) ... the strings of image stored path
                 null_img_num (torch.torch(1D class ID)) ... the number of blank images
        """
        imgs_transformed, label, label_id, dir_path, null_img_num = self._pull_item(index)
        return imgs_transformed, label, label_id, dir_path, null_img_num

    def _pull_item(self, index):
        """
        The implementation of the method __getitem__()

        @param index:
        @return:
        """

        # 1. Set directories to load raw-image
        dir_path = self.video_list[index]  # 画像が格納されたフォルダ
        indices = self._get_eq_spaced_indices(dir_path)  # 読み込む画像idxを求める
        # 1.5 Loading image while counting the number of blank images
        img_group, null_img_num = self._load_imgs(
            dir_path, self.img_tmpl, indices)  # リストに読み込む
        # 2.Get label / label_id
        label = dir_path.split(self.path_delim)[self.slash_sp_num]  # 注意：windowsOSの場合
        label_id = self.label_id_dict[label]  # idを取得
        # 3. Run Pre-process to image tensor
        imgs_transformed = self.transform(img_group, phase=self.phase)

        return imgs_transformed, label, label_id, dir_path, null_img_num

    def _load_imgs(self, dir_path, img_tmpl, indices):
        """
        Loading image files (which name must be defined referring to img_tmpl) stored in dir_path.
        The IDs to read will follow to indices

        @param dir_path:
        @param img_tmpl:
        @param indices:
        @return: Tupple of (1) Image Tensor that shape [L, CH, H, W], and (2) The num. of blank images
        """
        null_img_num = 0

        # Get images's filepath to load
        filepath = self._get_filepath_lst(dir_path, img_tmpl, indices)
        # Count psuedo image num
        for elem in filepath:
            if "[pad]" in elem:
                null_img_num += 1
            else:
                pass
        # now_ = time.time()
        # Single Process
        img_group = [_load_psuedo_true_img(file) for file in filepath]
        # Multiprocess
        # p = Pool(self.cpu_thread)  #
        # img_group = p.map(_load_psuedo_true_img, out_)
        # p.close()
        # print('Elapsed : ', time.time() - now_)
        return img_group, null_img_num

    def _get_filepath_lst(self, dir_path, img_tmpl, indices):
        """
        Create the list the element of which is image file path to load.
        If it's impossible, the element become "[pad]<h>x<w>x<ch>".
        All <h>, <w>, <ch> are used for blank image creation

        @param dir_path:
        @param img_tmpl:
        @param indices:
        @return:
        """
        ret = []
        for idx in indices:
            if idx > 0:
                ret.append(os.path.join(dir_path, img_tmpl.format(idx)))
            else:
                size_ = self.transform.size
                ret.append("[pad]" + str(size_) + "x" + str(size_) + "x" + str(
                    3))  # showing that [h, w, ch] for zero-tiled image
            pass
        return ret

    def _get_eq_spaced_indices(self, dir_path):
        """
        Make indices of loading image as List
        (e.g.) return = [strt_index, strt_index+1*seg_span, ..., strt_index+(seg_num-1)*seg_span].
        All variables such as  strt_index, seg_span and seg_num are set by this class constructor

        If element is not exist,  ID is set as "-1" and create pseudo image by np.zeros()

        @param dir_path:
        @return: indices as np.array.astype(np.int32)
        """
        file_list = os.listdir(path=dir_path)
        num_frames = len(file_list)
        indices = np.arange(self.strt_index, stop=num_frames - 1,
                            step=self.seg_span)  # [strt, strt+1*seg_span, strt+ 2*seg_span, ... , ]
        indices = indices[0:self.seg_num] if len(indices) > self.seg_num else indices
        indices = np.append(indices,
                            ([-1 for _ in range(self.seg_num - len(indices))]))  # idx -1 always means <PAD> image
        return indices.astype(np.int32)
