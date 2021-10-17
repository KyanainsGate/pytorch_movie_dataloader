# Example for multi_segment mode
import os

import torch.utils.data

from video_annotation_dataset import VideoDatasetWithAnnotation
from transform_utils import VideoTransform
from method_utils import get_c2i_i2c_from_dir_hrc, get_mean_and_std, save_input_animation, \
    make_anno_list, MultiSegmentVideoList, make_datapath_list

if __name__ == '__main__':
    # vieo_listの作成
    root_path = os.path.join('data', 'tsu_frames')
    anno_root_path = os.path.join('data', 'tsu_annotations')
    video_list = make_datapath_list(root_path)

    anno_list = make_anno_list(anno_root_path)
    annotated_classes = ["N/A", "PAD", "Enter", "Walk", "Sit_down", "Put_something_on_table", "Use_tablet"]

    videolist_with_path = MultiSegmentVideoList(root_path, seg_span=48, seg_len=16)
    top_images, top_file2indices = videolist_with_path()
    total_frames = videolist_with_path.total_frames

    cls2id, _ = get_c2i_i2c_from_dir_hrc(root_path)

    # Instance of transforms to read
    resize, crop_size = 224, 224
    mean, std = [54, 60, 64], [34, 31, 31]
    video_transform = VideoTransform(resize, crop_size, mean, std)

    # Instance of torch.utils.data.Dataset
    val_dataset = VideoDatasetWithAnnotation(
        top_images,
        # video_list,
        anno_list,
        video_frames=total_frames,
        annotated_classes=annotated_classes,
        label_id_dict=cls2id,
        seg_span=48,
        seg_len=16,
        phase="val",
        transform=video_transform,
        slash_sp_num=2,
        img_tmpl='image_{:05d}.jpg',
        multi_segment=True,
        top_file2indices=top_file2indices
    )
    #
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
    print("elem[5]:", batch[5])

    print('\n %%% Dataset analysis %%%')
    get_mean_and_std(val_dataloader)  # Get mean and std of dataset

    print('\n %%% Show processed image %%%')
    save_input_animation(batch[0], batch_id=0)  # Loading image example is revealed
