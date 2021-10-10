# -*- coding: utf-8 -*-

# Developed based on following source code
# https://github.com/YutaroOgawa/pytorch_advanced/blob/master/9_video_classification_eco/9-4_2_convert_mp4_to_jpeg.ipynb

import os
import subprocess  # ターミナルで実行するコマンドを実行できる
import argparse
import pandas as pd

if __name__ == '__main__':
    description = 'Create separated image.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('from_path', type=str, )
    p.add_argument('to_path', type=str, )
    p.add_argument('--scale', type=str, default="-1:256")
    p.add_argument('--movie_ext', type=str, default=".mp4")
    args = p.parse_args()

    # 動画が保存されたフォルダ「kinetics_videos」にある、クラスの種類とパスを取得
    from_path = args.from_path
    to_path = args.to_path
    movie_ext = args.movie_ext
    scale_ = args.scale
    class_list = os.listdir(from_path)

    # 各クラスの動画ファイルを画像ファイルに変換する
    for class_list_i in (class_list):  # クラスごとのループ

        # クラスのフォルダへのパスを取得
        class_from_path = os.path.join(from_path, class_list_i)
        class_to_path = os.path.join(to_path, class_list_i)

        # 各クラスのフォルダ内の動画ファイルをひとつずつ処理するループ
        for file_name in os.listdir(class_from_path):

            # ファイル名と拡張子に分割
            name, ext = os.path.splitext(file_name)

            # mp4ファイルでない、フォルダなどは処理しない
            if ext != movie_ext:
                continue

            # 動画ファイルを画像に分割して保存するフォルダ名を取得
            dst_directory_path = os.path.join(class_to_path, name)

            # 上記の画像保存フォルダがなければ作成
            if not os.path.exists(dst_directory_path):
                os.makedirs(dst_directory_path)

            # 動画ファイルへのパスを取得
            video_file_path = os.path.join(class_from_path, file_name)

            # ffmpegを実行させ、動画ファイルをjpgにする （高さは256ピクセルで幅はアスペクト比を変えない）
            # kineticsの動画の場合10秒になっており、大体300ファイルになる（30 frames /sec）
            cmd = 'ffmpeg -i \"{}\" -vf scale={} \"{}/image_%05d.jpg\"'.format(
                video_file_path, scale_, dst_directory_path)
            print("[DEBUG]", cmd)
            subprocess.call(cmd, shell=True)

    print("動画ファイルを画像ファイルに変換しました。")

    pass
