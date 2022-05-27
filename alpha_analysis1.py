import numpy as np
from matplotlib import pyplot as plt
import cv2
from matplotlib import animation

# 打开交互模式
plt.ion()
fig1 = plt.figure('frame')
# fig2 = plt.figure('subImg')
src = cv2.imread('imgs/3_src_resized.jpg')
src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
up = 14
left = 12
T_height = 181
T_width = 259
src_cropped = src[up: up+T_height, left: left+T_width].copy()
src_cropped = src_cropped.astype(np.uint8)
print(src_cropped.shape)
for i in range(1, 11):
	# 进行自己的处理
    if i > 1:
        old_a = a
    a = np.load(f'saved_npy/save{i}.npy')
    print(a.shape)
    masked_src = cv2.bitwise_or(src_cropped, np.zeros_like(src_cropped), mask=a)
    if i > 1:
        print(np.sum(np.abs(old_a - a)))
	#--------动态显示----------#
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.axis('off')  # 关掉坐标轴
    ax1.imshow(masked_src)
    # ax1.plot(p1[:, 0], p1[:, 1], 'g.')
	#停顿时间
    plt.pause(0.6)
	#清除当前画布
    fig1.clf()

plt.ioff()
