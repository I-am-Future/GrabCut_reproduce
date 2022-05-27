import numpy as np
from matplotlib import pyplot as plt
import cv2
from matplotlib import animation

# 打开交互模式
plt.ion()
fig1 = plt.figure('frame')
# fig2 = plt.figure('subImg')
src = cv2.imread('imgs/1_src_resized.jpg')
src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
up = 12
left = 11
T_height = 180
T_width = 265
for i in range(1, 21):
	# 进行自己的处理
    a = np.load(f'saved_npy/save{i}.npy')
    print(a.shape)
    masked_src = cv2.bitwise_or(src, np.zeros_like(src), mask=a)
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
