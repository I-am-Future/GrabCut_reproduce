# grabcut_utils_fromcv2.py
# My reproduction to the GrubCut with OpenCV-C++'s source code. 

import cv2
import numpy as np
from sklearn.cluster import KMeans

class GMM():
    def __init__(self) -> None:
        ''' A full-covariance Gaussian mixture model implementation. '''
        self.numComponents = 5
        self.dim = 3
        # mixing coefficient
        self.mixCoef = np.zeros((self.numComponents, ), dtype=np.float64)
        # mu
        self.mu = np.zeros((self.numComponents, self.dim), dtype=np.float64)
        # sigma
        self.sigma = np.zeros((self.numComponents, self.dim, self.dim), dtype=np.float64)
        # inv_sigma, easy for computing
        self.inv_sigma = np.zeros((self.numComponents, self.dim, self.dim), dtype=np.float64)
        # det_sigma, easy for computing
        self.det_sigma = np.zeros((self.numComponents, ), dtype=np.float64)
        # sums, prods, count, total_count, for learning param
        self.sums = np.zeros((self.numComponents, self.dim), dtype=np.float64)
        self.prods = np.zeros((self.numComponents, self.dim, self.dim), dtype=np.float64)
        self.count = np.zeros((self.numComponents, ), dtype=np.float64)
        self.total_count = 0
        
    def update_conv_det(self, k):
        ''' Update the kth easy-for-computing variables. '''
        if self.mixCoef[k] > 0:
            self.det_sigma[k] = np.linalg.det(self.sigma[k])
            assert (self.det_sigma[k] > 1e-7), f"det of sigma[{k}] is too close to 0!"
            self.inv_sigma[k] = np.linalg.inv(self.sigma[k])

    def init_param(self, datas: np.ndarray) -> None:
        ''' Init the GMM parameters. datas: size[N, 3]. '''
        kmeans = KMeans(n_clusters=self.numComponents)
        clusters = kmeans.fit_predict(datas)
        print(clusters)
        self.init_learning()
        for i in range(len(datas)):
            self.add_sample(datas[i], clusters[i])
        self.end_learning()

    def forward_sum(self, x: np.ndarray) -> float:
        ''' Forward the x with all components in this GM model. '''
        assert( len(x) == self.dim )
        result = 0
        for i in range(self.numComponents):
            result += self.mixCoef[i] * self.forward_one(x, i)
        return result

    def forward_one(self, x: np.ndarray, k: int = -1) -> float:
        ''' Forward the x with one component in this GM model. k={1,2,3,4,5}. '''
        if self.mixCoef[k] > 0:
            dx = x - self.mu[k]
            return float(np.exp(-0.5 * dx @ self.inv_sigma[k] @ dx) / np.sqrt(self.det_sigma[k]))
        else: 
            return 0

    def forward_arr(self, x: np.ndarray) -> float:
        ''' Forward the x seperately in array in this GM model. '''
        assert( len(x) == self.dim )
        result = np.zeros((self.numComponents, ))
        for i in range(self.numComponents):
            result[i] = self.forward_one(x, i)
        return result

    def classify(self, x: np.ndarray) -> int:
        ''' Classify which component does x in (maximum prob). '''
        return np.argmax(self.forward_arr(x))
        
    def init_learning(self):
        ''' Init the help param in learning. '''
        self.sums = np.zeros((self.numComponents, self.dim), dtype=np.float64)
        self.prods = np.zeros((self.numComponents, self.dim, self.dim), dtype=np.float64)
        self.count = np.zeros((self.numComponents, ), dtype=np.float64)
        self.total_count = 0

    def add_sample(self, x: np.ndarray, k: int):
        ''' Add samples and update the help param in learning. '''
        self.sums[k] += x
        self.prods[k][0][0] += x[0]*x[0]; self.prods[k][0][1] += x[0]*x[1]; self.prods[k][0][2] += x[0]*x[2]
        self.prods[k][1][0] += x[1]*x[0]; self.prods[k][1][1] += x[1]*x[1]; self.prods[k][1][2] += x[1]*x[2]
        self.prods[k][2][0] += x[2]*x[0]; self.prods[k][2][1] += x[2]*x[1]; self.prods[k][2][2] += x[2]*x[2]
        self.count[k] += 1
        self.total_count += 1

    def end_learning(self):
        ''' End the help param in learning. '''
        for k in range(self.numComponents):
            n = self.count[k]
            if n == 0:
                self.mixCoef[k] = 0
            else:
                self.mixCoef[k] = self.count[k] / self.total_count
                self.mu[k] = self.sums[k] / n

                self.sigma[k][0][0] = self.prods[k][0][0]/n - self.mu[k][0]*self.mu[k][0]
                self.sigma[k][0][1] = self.prods[k][0][1]/n - self.mu[k][0]*self.mu[k][1]
                self.sigma[k][0][2] = self.prods[k][0][2]/n - self.mu[k][0]*self.mu[k][2]
                self.sigma[k][1][0] = self.prods[k][1][0]/n - self.mu[k][1]*self.mu[k][0]
                self.sigma[k][1][1] = self.prods[k][1][1]/n - self.mu[k][1]*self.mu[k][1]
                self.sigma[k][1][2] = self.prods[k][1][2]/n - self.mu[k][1]*self.mu[k][2]
                self.sigma[k][2][0] = self.prods[k][2][0]/n - self.mu[k][2]*self.mu[k][0]
                self.sigma[k][2][1] = self.prods[k][2][1]/n - self.mu[k][2]*self.mu[k][1]
                self.sigma[k][2][2] = self.prods[k][2][2]/n - self.mu[k][2]*self.mu[k][2]
                
                if np.linalg.det(self.sigma[k]) < 1e-7:
                    self.sigma[k] += np.diag([1e-3]*3)
                self.update_conv_det(k)



def calc_beta(img: np.ndarray) -> float:
    ''' '''
    height, width = img.shape[0], img.shape[1]
    beta = 0
    count = 0
    for h in range(height):  # 0 to height-1
        for w in range(width):
            for n_h, n_w in get_neighbors(h, w, height-1, width-1):
                beta += (img[h, w]-img[n_h, n_w])@(img[h, w]-img[n_h, n_w])
                count += 1
    return 1 / (2 * beta / count)


def select_pixels(src: np.ndarray, mat: np.ndarray, sel: int):
    ''' select pixels for GMM according to `mat`, return [N, 3] '''
    assert( sel == 0 or sel == 1 )
    idx = (mat == sel)
    selected = src[idx]
    return selected.reshape(-1, 3)


def get_neighbors(h: int, w: int, max_h: int, max_w: int) -> list:
    ''' Get 8-direction neighbors from the given index. '''
    result = []
    if h > 0 and w < max_w:  # up right
        result.append((h-1, w+1)) 
    if w < max_w:  # right
        result.append((h, w+1))
    if h < max_h and w < max_w:
        result.append((h+1, w+1))
    if h < max_h:
        result.append((h+1, w))
    return result


def get_nlink(p_h: int, p_w: int, q_h: int, q_w: int, 
        src: np.ndarray, gamma: float, beta: float):
    ''' get the n-link value. NOTE: p, q are in global coordinates! '''
    diff = src[p_h, p_w] - src[q_h, q_w]
    return gamma * np.exp(-beta * np.dot(diff, diff)) / ((p_h-q_h)**2 + (p_w-q_w)**2)**0.5


def visualization(img: np.ndarray, alpha: np.ndarray) -> None:
    ''' Visualization of GrabCut (using `alpha` on image `img`). '''
    masked_img = cv2.bitwise_or(img, np.zeros_like(img), mask=alpha)
    cv2.imshow('Output', masked_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class GrabcutInfo():
    def __init__(self, bkgmodel, fkgmodel, alpha, trimap, g, l, b) -> None:
        ''' An encapsulated class for passing the GrabCut informations. '''
        self.bkgmodel = bkgmodel
        self.fkgmodel = fkgmodel
        self.alpha = alpha
        self.trimap = trimap
        self.gamma = g
        self.beta = b
        self.lambda_ = l
