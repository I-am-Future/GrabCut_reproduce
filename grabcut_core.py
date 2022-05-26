import cv2
import numpy as np
from grabcut_utils import *
import maxflow

def GrabCut_kernel(
    src: np.ndarray,  # the input image
    n_epoches: int = 10, 
) -> np.ndarray:
    ''' '''
    height, width = src.shape[0], src.shape[1]
    ## initialization
    # Trimap: 0: Background, 1: Undefined, 2: Foreground
    Trimap = np.zeros((height, width), dtype=np.uint8) 
    up = 12
    down = -5
    left = 12
    right = -30
    T_height = height - up + down
    T_width = width - left + right
    Trimap[up: down, left: right] = 1
    # print(Trimap)
    # Alpha: 0: Background, 1: Foreground
    Alpha = Trimap.copy()
    # print(Alpha)
    # beta: 
    term_diff = []
    for h in range(height):  # 0 to height-1
        for w in range(width):
            for n_h, n_w in get_neighbors(h, w, height-1, width-1):
                term_diff.append(np.linalg.norm(src[h, w] - src[n_h, n_w])**2)
    beta = 1 / (2 * sum(term_diff) / len(term_diff))
    print(beta)
    # Init GMMs
    src = src.astype(np.float32)
    src_cropped = src[up: down, left: right].copy()
    print(src.shape)
    print(src_cropped.shape)
    print(src.dtype)
    GMM_back = GMM(5, 3)
    back_pixels = select_pixels(src, Alpha, 0)
    GMM_back.init_param(back_pixels/255)
    GMM_fore = GMM(5, 3)
    fore_pixels = select_pixels(src, Alpha, 1)
    GMM_fore.init_param(fore_pixels/255)    
    import pickle
    pickle.dump(GMM_back, open('gmmb.bin', 'wb'))
    pickle.dump(GMM_fore, open('gmmf.bin', 'wb'))
    GMM_back = pickle.load(open('gmmb.bin', 'rb'))
    GMM_fore = pickle.load(open('gmmf.bin', 'rb'))
    # print(GMM_back.mu)
    # print(GMM_back.mixCoef)
    for epoch in range(n_epoches):
        ## step 1: Assign GMM components to pixels
        k_vec = np.ones_like(Trimap) * -1  # k vector
        for h in range(height):
            for w in range(width):
                if Trimap[h, w] == 1:  # in T_u
                    if Alpha[h, w] == 1:
                        D_n = GMM_fore.get_dn(src[h, w]/255)
                    else:  # == 0
                        D_n = GMM_back.get_dn(src[h, w]/255)
                    index = np.argmin(D_n)
                    k_vec[h, w] = index
        # check use
        # print('finished 1')
        # for i in range(5):  print(np.sum(k_vec == i))
        ## step 2: Learn GMM parameters
        for a in range(2):
            for k in range(5):
                pixels = src[ (Alpha==a) & (k_vec==k) ] / 255  # shape: m * 3
                if len(pixels) == 0:
                    continue
                if a == 0:  # change on back GMM
                    GMM_back.mu[k] = np.mean(pixels, axis=0)
                    GMM_back.sigma[k] = np.cov(pixels, rowvar=False)
                else:  # a == 1
                    GMM_fore.mu[k] = np.mean(pixels, axis=0)
                    GMM_fore.sigma[k] = np.cov(pixels, rowvar=False)
                GMM_fore.mixCoef[k] = len(pixels) / np.sum(Alpha==a)
        # print(GMM_back.mu)   
        # print(GMM_back.mixCoef)     
        # print('finished 2')
        ## step 3: Estimate segmentation
        g = maxflow.Graph[float]()
        nodeids = g.add_grid_nodes((T_height, T_width))
        structure = np.array([[0, 0, 0],
                              [0, 0, 1],
                              [1, 1, 1]])
        for t_h in range(T_height):  # 0 to T_height-1
            for t_w in range(T_width):
                for n_h, n_w in get_neighbors(t_h, t_w, T_height-1, T_width-1):
                    # Add neighbor edges
                    nvalue = get_nlink(t_h+up, t_w+left, n_h+up, n_w+left, src/255, Alpha, 50, beta)
                    g.add_edge(nodeids[t_h, t_w], nodeids[n_h, n_w], nvalue, nvalue)
                    # Add s edge and t edge
                    slink = get_stlink(src_cropped[t_h, t_w]/255, GMM_back)
                    tlink = get_stlink(src_cropped[t_h, t_w]/255, GMM_fore)
                    # print(nvalue, slink, tlink)
                    g.add_tedge(nodeids[t_h, t_w], slink, tlink)
        flow = g.maxflow()
        print(f"Maximum flow: {flow}")
        new_alpha = np.zeros((T_height, T_width), np.uint8)
        for t_h in range(T_height):  # 0 to T_height-1
            for t_w in range(T_width):
                new_alpha[t_h, t_w] = g.get_segment(nodeids[t_h, t_w])
        print(new_alpha)  # 0: fore, 1: back
        np.save(f'saved_npy/save{epoch+1}.npy', new_alpha)
        # print('finished 3')
        Alpha[up: down, left: right] = 1-new_alpha
        print(f'epoch {epoch+1} of {n_epoches}')
    exit()

img = cv2.imread('imgs/1_src_resized.jpg')
print(img.shape)

GrabCut_kernel(img)


