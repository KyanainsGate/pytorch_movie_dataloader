import os

import torch.utils.data

from video_dataset import VideoDataset
from transform_utils import VideoTransform
from method_utils import make_datapath_list, get_c2i_i2c_from_dir_hrc, get_mean_and_std, save_input_animation

if __name__ == '__main__':
    # video_listの作成
    root_path = os.path.join('data', 'kinetics_images')
    video_list = make_datapath_list(root_path)
    cls2id, _ = get_c2i_i2c_from_dir_hrc(root_path)

    # Instance of transforms to read
    resize, crop_size = 224, 224
    mean, std = [104, 117, 123], [1, 1, 1]
    video_transform = VideoTransform(resize, crop_size, mean, std)

    # Instance of torch.utils.data.Dataset
    val_dataset = VideoDataset(video_list,
                               label_id_dict=cls2id,
                               seg_span=5,
                               seg_len=16,
                               phase="val",
                               transform=video_transform,
                               slash_sp_num=2,
                               img_tmpl='image_{:05d}.jpg')

    # Instance of torch.utils.data.DataLoader
    batch_size = 3
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
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
    # show_img(batch[0], batch_id=0, seq_id=2)  # Loading image example is revealed
    save_input_animation(batch[0], batch_id=1)
