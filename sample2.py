# Example for multi_segment mode
import os

import torch.utils.data

from video_dataset import VideoDataset
from transform_utils import VideoTransform
from method_utils import make_datapath_list, get_c2i_i2c_from_dir_hrc, show_img, get_mean_and_std, MultiSegmentVideoList

if __name__ == '__main__':
    # vieo_listの作成
    root_path = os.path.join('data', 'kinetics_images')

    videolist_with_path = MultiSegmentVideoList(root_path, seg_span=5, seg_len=16, strt_index=3)
    top_images, top_file2indices = videolist_with_path()
    cls2id, _ = get_c2i_i2c_from_dir_hrc(root_path)

    # Instance of transforms to read
    resize, crop_size = 224, 224
    mean, std = [75, 73, 69], [1, 1, 1]
    video_transform = VideoTransform(resize, crop_size, mean, std)

    # Instance of torch.utils.data.Dataset
    val_dataset = VideoDataset(top_images,
                               label_id_dict=cls2id,
                               seg_span=2,
                               seg_len=5,
                               phase="val",
                               transform=video_transform,
                               slash_sp_num=2,
                               img_tmpl='image_{:05d}.jpg',
                               multi_segment=True,
                               top_file2indices=top_file2indices
                               )

    # Instance of torch.utils.data.DataLoader
    batch_size = 3
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=2)

    # Batch outputs
    print('%%% Batch outputs %%%')
    batch_iterator = iter(val_dataloader)  # イテレータに変換
    batch = next(batch_iterator)  # 1番目の要素を取り出す
    print("elem[0]:", batch[0].shape)
    print("elem[1]:", batch[1])
    print("elem[2]:", batch[2])
    print("elem[3]:", batch[3])
    print("elem[4]:", batch[4])

    print('\n %%% Dataset analysis %%%')
    get_mean_and_std(val_dataloader)  # Get mean and std of dataset

    print('\n %%% Show processed image %%%')
    show_img(batch[0], batch_id=0, seq_id=2)  # Loading image example is revealed
