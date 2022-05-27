import cv2
import numpy as np
from grabcut_utils import get_neighbors
import cvxpy as cp

Alpha = np.load('saved_npy/1_mask.npy')
img = cv2.imread('imgs/1_src_resized.jpg')
print(Alpha.shape)
height, width = Alpha.shape
Alpha = Alpha.astype(np.uint8)
contours, hierarchy = cv2.findContours(Alpha.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)#查找轮廓
print(len(contours))
x = 0
for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    if area>10000:
        print(area)
        x = i
cnt = contours[x]
approx1 = cv2.approxPolyDP(cnt,1,True) #拟合精确度
trimap = Alpha.copy().astype(np.uint8) * 255

# cv2.imshow('alpha', Alpha*255)
# cv2.waitKey(0)

# img = cv2.polylines(img, [approx1], True, (128, ), 11)
# img = cv2.polylines(img, [approx1], True, (255, ), 1)
# cv2.imshow('approxPolyDP', img)
trimap = cv2.polylines(trimap, [approx1], True, (128, ), 6)
# trimap = cv2.polylines(trimap, [approx1], True, (255, ), 1)
# cv2.imshow('approxPolyDP1',trimap)
# cv2.waitKey(0)

allNodes = np.zeros_like(trimap)
allNodes = cv2.polylines(allNodes, [approx1], True, (255, ), 6)
# cv2.imshow('approxPolyDP2',allNodes)
# cv2.waitKey(0)


border = np.zeros_like(trimap)
border = cv2.polylines(border, [approx1], True, (255, ), 6)
border = cv2.polylines(border, [approx1], True, (0, ), 4)
# cv2.imshow('approxPolyDP2',border)
# cv2.waitKey(0)

mask = np.zeros_like(trimap)
mask = cv2.fillPoly(mask, [approx1], (255, ))
# cv2.imshow('approxPolyDP2',mask)
# cv2.waitKey(0)

in_border = cv2.bitwise_or(border, np.zeros_like(border), mask=mask)
out_border = cv2.bitwise_or(border, np.zeros_like(border), mask=255-mask)
inner = cv2.bitwise_xor(in_border, allNodes)
inner = cv2.bitwise_xor(inner, out_border)
# cv2.imshow('approxPolyDP2', in_border)
# cv2.waitKey(0)
# cv2.imshow('approxPolyDP3', out_border)
# cv2.waitKey(0)
# cv2.imshow('approxPolyDP4', inner)
# cv2.waitKey(0)


outNodes = []
inNodes = []
innerNodes = []
# construct
for h in range(height):
    for w in range(width):
        if out_border[h, w] > 0:  # is out (t)
            outNodes.append((h, w))
        elif in_border[h, w] > 0:  # is in (s)
            inNodes.append((h, w))
        elif inner[h, w] > 0:  # is inner (n)
            innerNodes.append((h, w))

# print(len(outNodes))
# print(len(inNodes))
# print(len(innerNodes))
nodes = innerNodes + inNodes + outNodes  # a list 
print(len(nodes))
nnedges = []
nsedges = []
ntedges = []

def get_cost(h1, w1, h2, w2, beta: float, eta: float):
    ''' '''
    result = (1-beta) * np.mean(img[h1, w1] - img[h2, w2])/255 + \
            beta * eta / (1 + abs(cv2.pointPolygonTest(cnt, ((h1+h2)/2, (w1+w2)/2), True)))
    return ((h1-h2)**2+(w1-w2)**2)**0.5 / (1 + float(result))
beta = 0.5
eta = 20
for (h, w) in innerNodes:
    for (n_h, n_w) in get_neighbors(h, w, height, width):
        if inner[n_h, n_w] > 0:  # n with n  
            nnedges.append((nodes.index((h, w)), nodes.index((n_h, n_w)), get_cost(h, w, n_h, n_w, beta, eta)))
            nnedges.append((nodes.index((n_h, n_w)), nodes.index((h, w)), get_cost(h, w, n_h, n_w, beta, eta)))
        elif in_border[n_h, n_w] > 0:  # n with s  
            nsedges.append((nodes.index((h, w)), nodes.index((n_h, n_w)), get_cost(h, w, n_h, n_w, beta, eta)))
        elif out_border[n_h, n_w] > 0:  # n with t  
            ntedges.append((nodes.index((h, w)), nodes.index((n_h, n_w)), get_cost(h, w, n_h, n_w, beta, eta)))
print()
# print(len(nnedges))
# print(len(nsedges))
# print(len(ntedges))
edges = nnedges + nsedges + ntedges
print(len(edges))
for i in range(5):
    print(nnedges[i])
    print(nsedges[i])
    print(ntedges[i])

y = cp.Variable(len(nodes))
z = cp.Variable(len(edges))

constraints = [z >= 0]
costs = []
# zij >= yi - yj
for idx, (nodeidx1, nodeidx2, cost) in enumerate(edges):
    if nodeidx2 >= len(innerNodes) + len(inNodes):  # out, [n, t]
        constraints += [z[idx] >= y[nodeidx1] - 0]
    elif nodeidx2 <= len(innerNodes):  # neighbor, [n, n]
        constraints += [z[idx] >= y[nodeidx1] - y[nodeidx2]]
    else:  # in, [n, s]
        constraints += [z[idx] >= 1 - y[nodeidx1]]
    costs.append(cost)

objective = cp.Minimize( cp.matmul(z, np.array(costs)) )
print('constraints:', len(constraints))

problem = cp.Problem(objective, constraints)

result = problem.solve()

print(y.value)
print(result)
np.save('saved_npy/s4_3.npy', y.value)
y = np.load('saved_npy/s4_3.npy')
print(y.shape)
# nodes: [(h, w)]
print(np.sum(y))
print(np.sum(y==0))
new_Alpha = Alpha.copy()
for i in range(len(y)):
    if y[i] > 0:
        new_Alpha[nodes[i][0], nodes[i][1]] = 0
    else:
        new_Alpha[nodes[i][0], nodes[i][1]] = 1

for h, w in outNodes:
    new_Alpha[h, w] = 0
for h, w in inNodes:
    new_Alpha[h, w] = 1

cv2.imshow('new_Alpha', new_Alpha * 255)

old_img = cv2.bitwise_or(img, np.zeros_like(img), mask=Alpha)
masked_img = cv2.bitwise_or(img, np.zeros_like(img), mask=new_Alpha)


cv2.imshow('final', old_img)
cv2.imshow('final1', masked_img)
cv2.waitKey(0)