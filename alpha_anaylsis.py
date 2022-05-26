import numpy as np
from matplotlib import pyplot as plt
import cv2
a = np.load('saved_npy/save10.npy')
src = cv2.imread('imgs/3_src_resized.jpg')
src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
up = 48
left = 7
T_height = 199 
T_width = 150
src_cropped = src[up: up+T_height, left: left+T_width].copy()
src_cropped = src_cropped.astype(np.uint8)

print(a.shape)
print(src_cropped.shape)
print(np.sum(a))
src_cropped = src_cropped.astype(np.uint8)
a = a.astype(np.uint8)
masked_src = cv2.bitwise_or(src_cropped, np.zeros_like(src_cropped), mask=a)
print(masked_src.shape)
plt.imshow(masked_src)
plt.show()

# plt.imsave('second_output.png', masked_src)