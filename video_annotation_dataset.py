import os

import pandas as pd
import numpy as np

from video_dataset import VideoDataset


class VideoDatasetWithAnnotation(VideoDataset):
    def __init__(self,
                 video_list,
                 anno_list,
                 annotated_classes,
                 video_frames,
                 label_id_dict,
                 seg_span,
                 seg_len,
                 phase,
                 transform,
                 slash_sp_num=1,
                 img_tmpl='image_{:05d}.jpg',
                 strt_index=1,
                 first_id_of_image_tmpl=1,
                 multi_segment=False,
                 top_file2indices=None,
                 ):
        """

        @param video_list:
        @param anno_list:
        @param annotated_classes:
        @param video_frames:
        @param label_id_dict:
        @param seg_span:
        @param seg_len:
        @param phase:
        @param transform:
        @param slash_sp_num:
        @param img_tmpl:
        @param strt_index:
        @param first_id_of_image_tmpl: Give img_tmpl, set to decide the ID-youngest image file name like 00001 or 00000
                                        Must be corresponding to top_file2indices[0,0]
        @param multi_segment:
        @param top_file2indices:
        """
        super().__init__(video_list, label_id_dict, seg_span, seg_len,
                         phase, transform, slash_sp_num,
                         img_tmpl,
                         strt_index,
                         multi_segment,
                         top_file2indices
                         )
        self.anno_list = anno_list
        self.img_dir2anno_dir = {k: v for k, v in
                                 zip((self.video_top_dirnames_witouhdip if os.path.isfile(
                                     video_list[0]) else video_list), anno_list)}
        self.annotated_classes = annotated_classes
        self.video_frames = video_frames
        self.video2anno = self._frame2cls_label(anno_list, self.video_frames, self.annotated_classes)
        self.first_id_of_image_tmpl = first_id_of_image_tmpl
        pass

    def __getitem__(self, index):
        """
        Get index ID when called by inter() and return all input's correspondence

        :param index:
        :return: transformed images ... (torch.torch.(5D tensor of [b, l, h, w, ch])),
                 label (list) ... the strings of labels
                 label_id (torch.torch(1D class ID)),
                 dir_path (list) ... the strings of image stored path
                 null_img_num (torch.torch(1D class ID)) ... the number of blank images
                 annotations :
        """
        imgs_transformed, label, label_id, dir_path, null_img_num = self._pull_item(index)
        annotations_id, _ = self._load_annotation(dir_path)
        return imgs_transformed, label, label_id, dir_path, null_img_num, annotations_id

    def _load_annotation(self, top_image_filepath):
        # 1. Load annotation filepath from directory structure
        match_annotation_filename = self.img_dir2anno_dir[
            os.path.dirname(top_image_filepath) if os.path.isfile(top_image_filepath) else top_image_filepath]
        # 2. the dictionary the Key of which is annotation_filepath and corresponding Value is ndarray
        #    of class label thorough time series. That is used for the table of class selection when given indices.
        class_labels_in_annofile = self.video2anno[match_annotation_filename]
        # 3. Indices of image frame number like [1, 11, ..., -1, -1]
        #    corresponding to images files such as [image0001.jpg, image00011.jpg, ..., PAD, PAD]
        # print(self.top_file2indices)
        if os.path.isfile(top_image_filepath):
            indices = self.top_file2indices[top_image_filepath]
        else:
            indices = self._get_eq_spaced_indices(top_image_filepath)
        # 4 Indices may include "-1" means "PAD". So to avoid confusion come from it,
        if not -1 in indices:
            # No padding included
            # TODO If ID the top image file start 0 (e.g. image0000.jpg, image00001.jpg, ... ), BUG may be shown
            id = class_labels_in_annofile[-1 * self.first_id_of_image_tmpl + indices]
        else:
            # Applied when padding is included
            must_be_replaced_by_pad = [elem == -1 for elem in indices]
            # TODO If ID the top image file start 0 (e.g. image0000.jpg, image00001.jpg, ... ), BUG may be shown
            id_tmp = class_labels_in_annofile[-1 * self.first_id_of_image_tmpl + indices]
            id = [int(id_tmp[i]) if (must_be_replaced_by_pad[i] == False) else self.annotated_classes.index("PAD") for i
                  in range(len(indices))]
            pass
        label = [self.annotated_classes[int(i)] for i in id]
        # print("[Debug] pooled annotation", top_image_filepath, " || img: ", indices, " || annotation: ",
        #       -1 * self.first_id_of_image_tmpl + indices)
        return np.array(id).astype(float), label

    def _get_clsid_from_annotation(self, annotation_filename: str, total_frmaes: int, class_list: list):
        """
        Create np.array the element of which is class labels come from class_list.

        @param annotation_filename: csv filename having the index of
            - "event" as class name
            - "start_frame" as event start frame, 1,2,3,4, ...,
            - "end_frame" as event end frame. 2,3,4, ...,
        @param total_frmaes: the total images, unsigned integer
        @param class_list: the list that all class are supposed to be defined
        @return: np.array of which element is correspond to class ID and the length is equal to total_frmaes.
        """
        # TODO Handling of duplicated labels. To avoid duplication,
        #  in this implementation class_id is overridden by following for loop
        #  (e.g.) if ID=2 and ID=3 happened at the same time, load label data shows ID=3 because 3 is larger.
        df = pd.read_csv(annotation_filename)
        frame2cls = np.zeros(total_frmaes)
        for idx in df.index.values:
            annotation = df.loc[idx]
            cls_id = class_list.index(annotation["event"])
            # np.array's index is [0,1,2,...], and annotation index is [1,2,3,...].
            # That's why, it is necessary "-1" for start_frame
            frame2cls[annotation["start_frame"] - 1:annotation["end_frame"]] = cls_id
            pass
        # print("[Debug] annotation ", annotation_filename, "len=", len(frame2cls), frame2cls[178:191])
        return frame2cls

    def _frame2cls_label(self, annotations: list, frames: list, classes: list):
        return {annotation: self._get_clsid_from_annotation(annotation, frame, classes) for annotation, frame in
                zip(annotations, frames)}
