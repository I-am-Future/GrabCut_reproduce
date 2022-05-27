# GrabCut_reproduce

A reproduce repository to the classic computer graphics paper: "'GrabCut': interactive foreground extraction using iterated graph cuts" (https://doi.org/10.1145/1015706.1015720)

In this repo, we have two version of GrabCut: the one fully reproduce from the original paper and the one which took references with `OpenCV`'s GrabCut C++ source code.  

The author reproduced one version (`grabcut_core_reproduce.py` and `grabcut_utils_reproduce.py`) fully on the original paper first, but the result is not satisfying. Then, the author took reference with the author took references with `OpenCV`'s GrabCut C++ source code and modify it into Python version code (`grabcut_core_fromcv2.py` and `grabcut_utils_fromcv2.py`) . The Demo shown below is with the `OpenCV` modification version. 

## Demo

### 1 

+ Brush notation (small area):

<img src="https://future-cos01-1312070282.cos.ap-guangzhou.myqcloud.com/%5Cphotos%5C1b.png" alt="1b" style="zoom: 80%;" />

+ Result: 

![Output_img1](https://future-cos01-1312070282.cos.ap-guangzhou.myqcloud.com/%5Cphotos%5COutput_img1.jpg)

### 2

+ Brush Notation (large area):

<img src="https://future-cos01-1312070282.cos.ap-guangzhou.myqcloud.com/%5Cphotos%5C3c.png" alt="3c" style="zoom:67%;" />

+ Result:

![Output_img3](https://future-cos01-1312070282.cos.ap-guangzhou.myqcloud.com/%5Cphotos%5COutput_img3.jpg)

## Requirements

- Programming Language: `python 3.x`
- Package Required: `numpy`, `opencv-python` (`cv2`), `sklearn`, `PyMaxflow`. 

 All packages above can be installed with `pip install <package name>`.


## Usage

For two versions of code, some utility functions are in `...utils...py` file, and core part is in `...core...py` file. In the `main.py` there is a demo. For the usage please use `fromcv2` version, which is faster (several to tens of seconds), and also has better segmentation quality and equips with an editing UI to manually set foreground and background after segmentation. 
