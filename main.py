import cv2

import grabcut_core


cropped_img = cv2.imread('imgs/1_src_cropped.jpg')
print(cropped_img.shape)

grabcut_core.GrabCut_kernel(cropped_img)

