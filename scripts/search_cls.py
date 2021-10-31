import sys
import pathlib
import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt

# Reload top directory path
__CURRENT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(os.path.dirname(str(__CURRENT_DIR)))

from method_utils import make_anno_list, MultiSegmentVideoList


def parser(feed_by_lst=None):
    """

    @param feed_by_lst:
    @return: (type: argparse.Namespace)
    """
    parser_ = argparse.ArgumentParser(description='Analysis annotations files')
    parser_.add_argument('image_root_path', type=str, help="Relative path to annotation root_path")
    parser_.add_argument('annotation_root_path', type=str, help="Relative path to annotation root_path")
    parser_.add_argument('--draw', action='store_true', help="Specify whether draw bar graph for visualization")
    parser_.add_argument('--draw_path', action='store_true',
                         help="The path to save drawed picture (Default: ./cls_histogram.png)",
                         default="cls_histogram.png")
    if feed_by_lst is not None:
        args = parser_.parse_args(feed_by_lst)
    else:
        args = parser_.parse_args()
        pass
    return args


def _read_and_search_cls(path_to_annotation, total_frames, void_class="N/A"):
    """
    Read annotation file named `path_to_annotation` and count the frame number of the annotated classes
    The frames don't assigned to annotated classes will be recognized as the class `void_class`

    @param path_to_annotation:
    @param total_frames:
    @param void_class:
    @return:
    """
    df = pd.read_csv(path_to_annotation)
    all_classes = sorted(set(set(df["event"].to_list())))
    cls2num = {key_class: 0 for key_class in all_classes}
    cls2num[void_class] = 0
    for idx in df.index.values:
        annotation = df.loc[idx]
        label = annotation["event"]
        cls2num[label] += annotation["end_frame"] - annotation["start_frame"] + 1
        pass
    cls2num[void_class] = total_frames - sum(cls2num.values())
    # print(cls2num)
    return cls2num


def _review_all_cls_distrib(anno2cls_distrib: dict):
    """
    Give the dictionary composed of the pair {filepath_to_csv: the dictionary of
    annotation class 2 frames like {"Walk": 32, ..., } },
    which is equivalent to the result of _read_and_search_cls(),
    output the summed up result of them

    @param anno2cls_distrib:
    @return:
    """
    ret_dict = {}
    distribs = list(anno2cls_distrib.values())
    for each_distrib in distribs:
        keys_in_each_distrib = list(each_distrib.keys())
        for key in keys_in_each_distrib:
            if key not in list(ret_dict.keys()):
                # As class label `key` didn't assigned to `ret_dict` before, define it in `ret_dict`
                ret_dict[key] = each_distrib[key]
            else:
                # Sum the frame numbers of `key` class to  `ret_dict`
                ret_dict[key] += each_distrib[key]
            pass
        pass
    # print(ret_dict)
    # print("Classes (", len(list(ret_dict.keys())), "classes ): ", ret_dict.keys(), )
    print("Classes: ", ret_dict.keys(), )
    print("Counts:  ", ret_dict.values())
    all_num = sum(list(ret_dict.values()))
    na_num = ret_dict['N/A']
    print("Total frames: {} / The class: `N/A` {} ({:.2f} %)".format(all_num, na_num, 100. * na_num / all_num))
    return ret_dict


def _draw_bar_graph(showing_dict: dict, out_png_path="cls_histogram.png"):
    """
    Draw the histogram picture through matplotlib

    @param showing_dict:
    @param out_png_path:
    @return:
    """
    # https://stackoverflow.com/questions/16010869/plot-a-bar-using-matplotlib-using-a-dictionary
    pd.DataFrame(showing_dict, index=['']).plot(kind='bar')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=9)
    plt.tight_layout()
    if out_png_path != "":
        plt.savefig(out_png_path)
        pass
    else:
        plt.show()
    pass


if __name__ == '__main__':
    args = parser()
    total_frames = MultiSegmentVideoList(root_path=args.image_root_path, seg_span=1, seg_len=1).total_frames
    anno_filepath = make_anno_list(args.annotation_root_path)
    anno2distrib = {anno_filepath[i]: _read_and_search_cls(anno_filepath[i], total_frames[i]) for i in
                    range(len(anno_filepath))}
    ret_dict = _review_all_cls_distrib(anno2distrib)
    _draw_bar_graph(ret_dict, args.draw_path) if args.draw else _draw_bar_graph(ret_dict, "")
    pass
