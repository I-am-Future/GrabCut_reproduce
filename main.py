import cv2

import grabcut_core_reproduce 
import grabcut_core_fromcv2


img = cv2.imread('imgs/2_src_resized.jpg')
print(img.shape)

left, up, t_width, t_height = cv2.selectROI('Select fkg', img, False, False )
cv2.destroyAllWindows()

alpha = grabcut_core_fromcv2.GrabCut_kernel(img, up, left, t_height, t_width, 5, True) # True means with editing UI

grabcut_core_fromcv2.visualization(img, alpha)

