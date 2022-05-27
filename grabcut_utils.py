import numpy as np
from sklearn.cluster import KMeans

class GMM():
    def __init__(self, K: int, dim: int = 3) -> None:
        ''' A full-covariance Gaussian mixture model implementation. 
            @param 
        '''
        self.numComponent = K
        self.dim = dim
        # mixing coefficient
        self.mixCoef = np.ones((K, ), dtype=np.float32) / self.numComponent
        # mu
        self.mu = np.random.randn(K, dim) + 0.5
        # sigma
        self.sigma = np.stack([np.diag(np.random.rand(dim)+0.25) for _ in range(K)])
        # self.sigma = np.stack([np.eye(self.dim) for _ in range(K)])
        # inv_sigma, easy for computing
        self.inv_sigma = np.zeros((K, dim, dim))
        # factor, easy for computing
        self.factor = np.zeros(K)
        self.update_()

    def update_(self):
        ''' Update the help variables. '''
        for i in range(self.numComponent):
            while True:
                try:
                    self.inv_sigma[i] = np.linalg.inv(self.sigma[i])
                    break
                except:
                    print('singular caught!')
                    self.sigma[i][0,0] += 1e-4
        for i in range(self.numComponent):
            self.factor[i] = (2.0 * np.pi)**(self.dim / 2.0) * (np.fabs(np.linalg.det(self.sigma[i])))**(0.5)

    def init_param(self, datas: np.ndarray) -> None:
        ''' datas: size[N, K]. '''
        kmeans = KMeans(n_clusters=self.numComponent)
        clusters = kmeans.fit_predict(datas)
        print(clusters)
        for k in range(self.numComponent):
            idx = (clusters == k)
            if len(idx) == 0:
                continue
            self.mu[k] = np.mean(datas[idx], axis=0)
            self.sigma[k] = np.cov(datas[idx], rowvar=False)
            self.mixCoef[k] = len(datas[idx]) / len(datas)
        self.update_()

        # for epoch in range(100):
        #     # E
        #     responses = np.zeros((self.numComponent, datas.shape[0]))
        #     for i, data in enumerate(datas):
        #         for k in range(self.numComponent):
        #             responses[k, i] = self.mixCoef[k] * self.__gaussian(data, k) 
        #     responses = responses / np.sum(responses,axis=0)
        #     # M
        #     old_mu = self.mu.copy()
        #     old_sigma = self.sigma.copy()
        #     feature_sum = np.sum(responses, axis=1)
        #     for i in range(self.numComponent):
        #         mu = np.dot(responses[i, :], datas) / feature_sum[i]
        #         sigma = np.zeros((self.dim, self.dim))
        #         for j in range(datas.shape[0]):
        #            sigma += responses[i, j] * np.outer(datas[j, :] - mu, datas[j, :] - mu)
        #         sigma = sigma / feature_sum[i]

        #         self.mu[i] = mu
        #         self.sigma[i] = sigma
        #         self.mixCoef[i] = feature_sum[i] / np.sum(feature_sum)
        #     self.update_()
        #     if (norm_ := np.linalg.norm(self.sigma - old_sigma).flatten()) < 1.5e-2:
        #         print(f"EM's sigma difference at {epoch} is {norm_} smaller than 1.5E-2, exit.")
        #         break
        #     print(f"epoch {epoch} / 100 in EM, {norm_}")
            
    def forward_sum(self, x: np.ndarray) -> float:
        ''' Forward (sum of all probs) of x in this GM model. '''
        assert( len(x) == self.dim )
        result = 0
        for i in range(self.numComponent):
            result += float(self.mixCoef[i] * self.__gaussian(x, i))
        return result

    def forward_sep(self, x: np.ndarray) -> float:
        ''' Forward (seperated probs) of x in this GM model. '''
        assert( len(x) == self.dim )
        result = np.zeros(self.numComponent)
        for i in range(self.numComponent):
            result[i] = self.__gaussian(x, i)
        return result

    def get_dn(self, x: np.ndarray) -> float:
        ''' Forward (seperated probs) of x in this GM model. '''
        assert( len(x) == self.dim )
        return -np.log(self.forward_sep(x)+1e-6) - np.log(self.mixCoef)
        
    def __gaussian(self, x: np.ndarray, k: int) -> float:
        ''' Utility function for k_th gaussian in this model. '''
        dx = x - self.mu[k]
        return np.exp(-0.5 * np.dot(np.dot(dx, self.inv_sigma[k]), dx)) / self.factor[k]


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

# print(get_neighbors(0, 0, 20, 20))
# print(get_neighbors(0, 10, 20, 20))
# print(get_neighbors(10, 20, 20, 20))
# print(get_neighbors(20, 20, 20, 20))

def get_nlink(p_h: int, p_w: int, q_h: int, q_w: int, 
        src: np.ndarray, Alpha: np.ndarray, gamma: float, beta: float):
    ''' get the n-link value. NOTE: p, q are in global coordinates! '''
    if Alpha[p_h, p_w] == Alpha[q_h, q_w]:
        return 0
    else:
        return gamma * np.exp(-beta * np.linalg.norm(src[p_h, p_w] - src[q_h, q_w]) ** 2)

def get_stlink(x: np.ndarray, model: GMM):
    ''' get s/t link depends on the model given. p is in global coordinates!'''
    return np.min(model.get_dn(x))

# gmm_back = GMM(5, 3)
# print(gmm_back.forward_sum(np.array([100,200,100])/255))
# print(gmm_back.forward_sep(np.array([100,200,100])/255))
