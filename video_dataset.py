# URL: https://github.com/KyanainsGate/pytorch_advanced/blob/master/9_video_classification_eco/utils/kinetics400_eco_dataloader.py

# 第9章 動画分類（ECO：Efficient 3DCNN）
# 9.4	Kinetics動画データセットからDataLoaderの作成

# 必要なパッケージのimport
import os

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
    def __init__(self, video_list, label_id_dict, seg_span, seg_len,
                 phase, transform, slash_sp_num=1,
                 img_tmpl='image_{:05d}.jpg',
                 strt_index=1,
                 multi_segment=False,
                 top_file2indices=None
                 ):
        """
        Constructor for building torch.Dataset

        @param video_list: a list of image-saved directories or image-paths. If image-paths, multi_segment must be True.
        @param label_id_dict: The dictionary for label to ID
        @param seg_span: Skip length to load image files per segment
        @param seg_len: Total length of segment
        @param phase: Set "train" or "val"
        @param transform: class the __call__() of which return torchvision.transforms.Compose() given phase
        @param slash_sp_num: The distance of class name labeled directory from the working directory
        @param img_tmpl: the image file template (Default: image_00001.jpg, image_00002.jpg, ....)
        @param strt_index: The index of top loading image. (e.g.) strt_index=3 means loading start from image_00003.jpg
        @param multi_segment: Switch multi_segment Mode(filepath are set as video_list) or NOT (directory names are set)
        @param top_file2indices: The dictionary of {element of video_list: indices of np.array, length==seg_len}.
               Must be set when multi_segment == True
        """
        self.video_list = video_list  # 動画画像のフォルダへのパスリスト
        self.label_id_dict = label_id_dict  # ラベル名をidに変換する辞書型変数
        self.seg_span = seg_span  # Define delta t of images
        self.seg_len = seg_len  # Fixed time series length to feed,
        self.phase = phase  # train or val
        self.transform = transform  # 前処理
        self.slash_sp_num = slash_sp_num
        self.img_tmpl = img_tmpl  # 読み込みたい画像のファイル名のテンプレート
        self.strt_index = strt_index
        self.null_img_num = 0
        self.top_file2indices = top_file2indices

        # Add for multi segment
        self.multi_segment = multi_segment  # loading multi segment mode
        if multi_segment:
            assert os.path.isfile(video_list[0]), "Set Top images as video_list "
            assert self.top_file2indices is not None, "Set video_list2indices"
            assert len(
                self.top_file2indices[
                    self.video_list[
                        0]]) == seg_len, "Match seg_len and the length of the value of Dictionary top_file2indices"
            assert self.top_file2indices[self.video_list[0]][1] - self.top_file2indices[self.video_list[0]][
                0] == self.seg_span, "Match seg_len to the step of the value of Dictionary top_file2indices"
            assert self.top_file2indices[self.video_list[0]][
                       0] == strt_index, "Match strt_index to the first element of top_file2indices's first Value"

            self._video_top_dirnames = [os.path.dirname(elem) for elem in self.video_list]
            self.video_top_dirnames_witouhdip = list(dict.fromkeys(self._video_top_dirnames))
            # print(self.video_top_dirnames_witouhdip)

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

    def _indices_from_topfile(self, top_image_filepath):
        indices = self.top_file2indices[top_image_filepath]
        return indices

    def _pull_item(self, index):
        """
        The implementation of the method __getitem__()

        @param index:
        @return:
        """

        if self.multi_segment:
            top_image_filepath = self.video_list[index]
            dir_path = os.path.dirname(top_image_filepath)  # 画像が格納されたフォルダ
            indices = self.top_file2indices[top_image_filepath]
            loading_key = top_image_filepath
        else:
            # 1. Set directories to load raw-image
            dir_path = self.video_list[index]  # 画像が格納されたフォルダ
            indices = self._get_eq_spaced_indices(dir_path)  # 読み込む画像idxを求める
            loading_key = dir_path
            pass
        # indices = self._get_eq_spaced_indices(dir_path)  # 読み込む画像idxを求める
        # 1.5 Loading image while counting the number of blank images
        img_group, null_img_num = self._load_imgs(
            dir_path, self.img_tmpl, indices)  # リストに読み込む
        # 2.Get label / label_id
        path_delim = "\\" if os.name == 'nt' else "/"  # Deliminator detection whether windows or not
        label = dir_path.split(path_delim)[self.slash_sp_num]
        label_id = self.label_id_dict[label]  # idを取得
        # 3. Run Pre-process to image tensor
        imgs_transformed = self.transform(img_group, phase=self.phase)

        return imgs_transformed, label, label_id, loading_key, null_img_num

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
        # print("[Debug] indices = filepath? ", indices, filepath)  # Comment-out to validate the indices and filepath
        # Count psuedo image num
        for elem in filepath:
            if "[pad]" in elem:
                null_img_num += 1
            else:
                pass
        # Single Process
        img_group = [_load_psuedo_true_img(file) for file in filepath]
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
        (e.g.) return = [strt_index, strt_index+1*seg_span, ..., strt_index+(seg_len-1)*seg_span].
        All variables such as  strt_index, seg_span and seg_len are set by this class constructor

        If element is not exist,  ID is set as "-1" and create pseudo image by np.zeros()

        @param dir_path:
        @return: indices as np.array.astype(np.int32)
        """
        file_list = os.listdir(path=dir_path if not os.path.isfile(dir_path) else os.path.dirname(dir_path))
        num_frames = len(file_list)
        indices = np.arange(self.strt_index, stop=num_frames - 1,
                            step=self.seg_span)  # [strt, strt+1*seg_span, strt+ 2*seg_span, ... , ]
        indices = indices[0:self.seg_len] if len(indices) > self.seg_len else indices
        indices = np.append(indices,
                            ([-1 for _ in range(self.seg_len - len(indices))]))  # idx -1 always means <PAD> image
        return indices.astype(np.int32)
