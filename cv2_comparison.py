import cv2 as cv
import numpy as np

src = cv.imread("imgs/1_src_resized.jpg")
r = cv.selectROI('input', src, False)  
roi = src[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

mask = np.zeros(src.shape[:2], dtype=np.uint8)
rect = (int(r[0]), int(r[1]), int(r[2]), int(r[3]))

bgdmodel = np.zeros((1,65),np.float64)
fgdmodel = np.zeros((1,65),np.float64)

cv.grabCut(src,mask,rect,bgdmodel,fgdmodel, 11, mode=cv.GC_INIT_WITH_RECT)

mask2 = np.where((mask==1) + (mask==3), 255, 0).astype('uint8')

print(mask2.shape)

result = cv.bitwise_and(src,src,mask=mask2)

cv.imshow('roi', roi)
cv.imshow("result", result)
cv.waitKey(0)
cv.destroyAllWindows()