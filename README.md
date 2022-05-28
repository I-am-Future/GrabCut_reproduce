# GrabCut_reproduce

A reproduce repository to the classic computer graphics paper: "'GrabCut': interactive foreground extraction using iterated graph cuts" (https://doi.org/10.1145/1015706.1015720)

In this repo, we have two version of GrabCut: 1 the one fully reproduce from the original paper and 2 the one which took references with `OpenCV`'s GrabCut C++ source code.  

+ paper version: (`grabcut_core_reproduce.py` and `grabcut_utils_reproduce.py`) 
  + Reproduction is more similar to the paper, basically followed the equations and algorithms on the paper. It has less performance and speed due to my poor coding ability. 
+ `OpenCV` version: (`grabcut_core_fromcv2.py` and `grabcut_utils_fromcv2.py`)
  + There are some difference to the original paper. I read the C++ source code and modified it for Python. It has better performance and is very fast. Meanwhile, a UI was designed for the users to perform further fore/background editing.  

## Demo

The Demo shown below is with the `OpenCV` modification version. 

### 1 

+ Brush notation (small area):

<img src="https://future-cos01-1312070282.cos.ap-guangzhou.myqcloud.com/%5Cphotos%5C1b.png" alt="1b" style="zoom: 80%;" />

+ Result: 

<img src="https://future-cos01-1312070282.cos.ap-guangzhou.myqcloud.com/%5Cphotos%5COutput_img1.jpg" alt="1b" style="zoom: 80%;" />


### 2

+ Brush Notation (large area):

<img src="https://future-cos01-1312070282.cos.ap-guangzhou.myqcloud.com/%5Cphotos%5C3c.png" alt="3c" style="zoom:67%;" />

+ Result:

<img src="https://future-cos01-1312070282.cos.ap-guangzhou.myqcloud.com/%5Cphotos%5COutput_img3.jpg" alt="1b" style="zoom: 80%;" />


## Requirements

- Programming Language: `python 3.x`
- Package Required: `numpy`, `opencv-python` (`cv2`), `sklearn`, `PyMaxflow`. 

 All packages above can be installed with `pip install <package name>`.


## Usage

For two versions of code, some utility functions are in `grabcut_utils_xxx.py` file, and core part is in `grabcut_utils_xxx.py` file. In the `main.py` there is a demo. For the usage, please use `fromcv2` version, which is faster (several to tens of seconds), and also has better segmentation quality, and equips with an editing UI to manually set foreground and background after segmentation. 
