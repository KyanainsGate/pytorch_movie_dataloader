# pytorch_movie_dataloader
- Loading movie data which was converted to image files and passing to the model as `torch.Tensor` type.
- Here is the sample code for [Kinetics 400](https://deepmind.com/research/open-source/kinetics).

## Environment
Tested on following configurations;
- Windows 11, and wsl2 on it
- Ubuntu 20.04
- `Python 3.8.10` with 
  - `torch==1.7.0` and `torchvision==0.8.1`

## Features
### Directory structure
Directory structure of raw-data must be designed as following;
- `root_path` is set in your source code.
- Each names and extensions of an image file (in following case `image_00002` and `.jpg` are correspond to them) can be designed freely
```
root_path 
    |-- classA
    |   |-- hoge
    |   |   |-- image_00001.jpg
    |   |   |-- image_00002.jpg
    |   |   :
    |   |   `-- image_00030.jpg                     
    |   `-- piyo
    |       |-- image_00001.jpg
    |       :
    |   
    |-- classB
    |   |-- hogegege
    |   |   |-- image_00001.jpg
    |   |   :
    :   :
```

### Automatic padding to non-readable image
- To match all tensor shape of sampled batches which are supposed to **same image size and frequency** and **different stream  length**, pseudo image data (like `[PAD]` in NLP) is automatically generated.
- Loading `[image_00002.jpg, image_00012.jpg, ... image_00032.jpg]` can be done by setting `strt_index=1`, `seg_span=5`, and `seg_len=4` in `VideoDataset()`  . 
- If the number of image files is fewer than requirements, non-readable image is replaced to blank image.
    - The above case, setting `start_index=1`, `seg_span=5`, and `seg_len=4` to read from `root path/class A/hoge` which contains 30 images seems to be fail because `image_00032.jpg` cannot be read. 
    -  To avoid those problem, `image_00032.jpg` is replaced pseudo image created by `np.zeros((h, w, ch)` in this implementation. 

### Batch outputs
Tuple of 4 elements, implemented in the class method `__getitem__()` in `VideoDataset()`
- elem[0]: `torch.Tensor` of images that shapes `[batch, sequence, ch, h, w]`
- elem[1]: Tuple of class labels
- elem[2]: `torch.Tensor` of class ID shapes `[batch, id]`
- elem[3]: Tuple of directories to read image files
- elem[4]: The number of blank image, unsigned integer

## Run sample
### Create virtual environment through conda
Run [download.py](video_download/download.py) with interpreter `Python 2.7` and some packages
```shell
conda create -n kinetics_loading python=2.7
conda activate kinetics_loading
pip install -r video_download/requirements_for_py27.txt # install thorough pip
```
### Download kinetics sample (for Linux OS)
Download a few sample (Taking a few minutes)
```shell
python ./video_download/download.py ./video_download/kinetics-400_val_4videos.csv ./data/kinetics_videos/
```
Create image files 
```shell
python ./video_download/separate_img.py ./data/kinetics_videos/ ./data/kinetics_images/
```
### See results (Both Windows and Linux)
Activate appropriate interpreter (Python 3.8 or more) and run;
```shell
python sample.py
```
- It shows the example of outputs the shape of which is mentioned in `Batch outputs`

In `sample.py`, since key image file names (which is instead of the base path of image files) are fed to the constructor `VideoDataset()` as `video_list`, the result of elem[3] and argument is different from `sample.py`. 
```shell
python sample2.py
```
- Use this if the number of image files are greater than  `seg_span * seg_len`
- Note that the first argument of `VideoDataset()` is different from `sample.py`
    - sample.py  ... the video directories like `[root_path/classA/hoge, root_path/classA/piyo, ..., ]`
    - sample2.py ... filenames like `[root_path/classA/hoge/image_00008.jpg, root_path/classA/piyo/image_00058.jpg, ..., ]`

## Annotation loading
### Overview
If the corresponding annotation files are also stored like following structure, class `VideoDatasetWithAnnotation()` supplies the annotation loading while loading image files;

```
root_path_of_annotation 
    |-- classA
    |   |-- hoge.txt
    |   `-- piyo.txt
    |   
    |-- classB
    |   |-- hogegege.txt
    |   |   :
    :   :
```

### Sample
The annotation format is reference to 
```shell
python sample_anno.py
```
- The annotations format is only adopted to [The Toyota Smarthome Untrimmed dataset](https://project.inria.fr/toyotasmarthome/)
- The 6th outputs of batch output (=`elem[5]:`) is class labels of each frame

## Support scripts
### Create image files for training
```shell
python video_download/separate_img.py <raw-videos> data/ --file_template image_%06d.jpg
```
The `<raw-videos>` structure must be set as followings; 
```shell
raw-videos
    |-- classA
    |   |-- hoge.mp4
    |   `-- piyo.mp4
    |   
    |-- classB
    |   |-- hogegege.mp4
    |   |   :
    :   :
```
### Analysis dataset invariance
[search_annotation_cls.py](scripts/search_annotation_cls.py) gives the number of the frames which is assigned to class and not assigned class as `N/A`
```shell
# (e.g.)
python scripts/search_annotation_cls.py  root_path root_path_of_annotation --draw
```


## Reference
- [torchvision.datasets.DatasetFolder](https://pytorch.org/vision/stable/datasets.html#base-classes-for-custom-datasets) for basic design of the directory structure 
- [pytorch_advanced
](https://github.com/YutaroOgawa/pytorch_advanced)
- Project page of [Toyota Smart Home](https://project.inria.fr/toyotasmarthome/)