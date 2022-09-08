## Installation
1\) Environment requirements
* Python 3.x
* Pytorch 1.7 or higher
* CUDA 9.2 or higher

Create a conda virtual environment and activate it.
```
conda create -n Fsnet python=3.7
conda activate Fsnet
```

2\) Clone the our project.
```
git clone https://github.com/Gary3410/shape_estimation.git
```

3\) Install the dependencies.
```
pip install matplotlib
pip install numpy
pip install opencv-contrib-python
pip install opencv-python
```
4\) Build CD_loss
```
cd Fsnet
cd chamer3D
python setup.py install
```
## Prepare Data

```
 链接：https://pan.baidu.com/s/1MCUWcxA5r7wf1hJf5f330A?pwd=gwdo
```
创建路径
```
cd Fsnet
mkdir data
```
数据文件直接放在data文件下
文件目录如下：
```
Fsnet
├── data
│   ├── box
│   │   ├──1
│   │   │  ├──0_depth.png
│   │   │  ├──0_label.pkl
│   │   │  ├──0_rgb.png
│   │   │  ├──0_seg.png
│   │   ├──points
│   │   │  ├──pose0000001.txt
│   │   │   ...
│   │   ├──points_labs
│   │   │  ├──lab0000001.txt
│   │   ├──box.ply
│   ├── can
│   │   ├──can.ply
│   │   ...
│   ├── mug
│   │   ├──mug.ply
│   │   ...
```
```
box, can, mug都是物体形状大类
box.ply, mug.ply都是物体模板
pose00000001.txt为采集的点云块
lab00000001.txt为标签点云(目前为点云中各个点的标签, 用于区分前景与背景)
1中文件为尺度标签, 后续不会使用
```
## Train
```
CUDA_VISIBLE_DEVICES=0 python train_test_cp.py
```
