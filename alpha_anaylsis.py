import numpy as np
from matplotlib import pyplot as plt
import cv2
a = np.load('saved_npy/save8.npy')
src = cv2.imread('imgs/1_src_resized.jpg')
src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
up = 12
down = -5
left = 12
right = -30
src_cropped = src[up: down, left: right].copy()

print(a.shape)
print(src_cropped.shape)
print(np.sum(a))
src_cropped = src_cropped.astype(np.uint8)
a = a.astype(np.uint8)
masked_src = cv2.bitwise_or(src_cropped, np.zeros_like(src_cropped), mask=1-a)
print(masked_src.shape)
plt.imshow(masked_src)
plt.show()

# plt.imsave('second_output.png', masked_src)