import cv2
import numpy as np
from grabcut_utils import *
import maxflow

def GrabCut_kernel(
    src: np.ndarray,  # the input image
    up: int, 
    left: int, 
    T_height: int, 
    T_width: int,
    n_epoches: int = 8, 
) -> np.ndarray:
    ''' '''
    print(up, left, T_height, T_width)
    height, width = src.shape[0], src.shape[1]
    ## initialization
    # Trimap: 0: Background, 1: Undefined, 2: Foreground
    Trimap = np.zeros((height, width), dtype=np.uint8) 
    Trimap[up: up+T_height, left: left+T_width] = 1
    # Alpha: 0: Background, 1: Foreground
    Alpha = Trimap.copy()
    # beta: 
    term_diff = []
    for h in range(height):  # 0 to height-1
        for w in range(width):
            for n_h, n_w in get_neighbors(h, w, height-1, width-1):
                term_diff.append(np.linalg.norm(src[h, w] - src[n_h, n_w])**2)
    beta = 1 / (2 * sum(term_diff) / len(term_diff))
    # print(beta)
    src = src.astype(np.float32)
    src_cropped = src[up: up+T_height, left: left+T_width].copy()
    print(src.shape)
    print(src_cropped.shape)
    print(src.dtype)
    # Init GMMs
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
        GMM_back.update_()
        GMM_fore.update_()
        ## step 3: Estimate segmentation
        g = maxflow.Graph[float]()
        nodeids = g.add_grid_nodes((T_height, T_width))
        for t_h in range(T_height):  # 0 to T_height-1
            for t_w in range(T_width):
                for n_h, n_w in get_neighbors(t_h, t_w, T_height-1, T_width-1):
                    # Add neighbor edges
                    g.add_edge(nodeids[t_h, t_w], nodeids[n_h, n_w], 
                            get_nlink(t_h+up, t_w+left, n_h+up, n_w+left, src/255, Alpha, 50, beta), 
                            get_nlink(t_h+up, t_w+left, n_h+up, n_w+left, src/255, Alpha, 50, beta))
                    # Add s edge and t edge
                    g.add_tedge(nodeids[t_h, t_w], 
                            get_stlink(src_cropped[t_h, t_w]/255, GMM_back), # slink
                            get_stlink(src_cropped[t_h, t_w]/255, GMM_fore))  # tlink
        flow = g.maxflow()
        print(f"Maximum flow: {flow}")
        new_alpha = np.zeros((T_height, T_width), np.uint8)
        for t_h in range(T_height):  # 0 to T_height-1
            for t_w in range(T_width):
                new_alpha[t_h, t_w] = g.get_segment(nodeids[t_h, t_w])
        new_alpha = 1 - new_alpha
        # print(new_alpha)  # 0: back, 1: fore
        np.save(f'saved_npy/save{epoch+1}.npy', new_alpha)
        Alpha[up: up+T_height, left: left+T_width] = new_alpha
        print(f'epoch {epoch+1} of {n_epoches}')

    ## User Editting
    src_cropped = src_cropped.astype(np.uint8)
    Alpha = Alpha.astype(np.uint8)
    def callback(event, x, y, flags, param):
        ''' Callback function of the window. '''
        nonlocal isForeBrush
        if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            if isForeBrush:
                cv2.circle(fb_matrix, (x, y), 5, (2, ), -1)
                cv2.circle(display_img, (x, y), 5, (255, 255, 255), -1)
            else:
                cv2.circle(fb_matrix, (x, y), 5, (0, ), -1)
                cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('Editing', display_img)
    while True:  # every whole iteration
        fb_matrix = np.ones((T_height, T_width), np.uint8)
        isForeBrush = True
        masked_src = cv2.bitwise_or(src_cropped, np.zeros_like(src_cropped), mask=Alpha[up: up+T_height, left: left+T_width])
        display_img = src_cropped.copy()
        cv2.imshow('Current', masked_src)
        cv2.namedWindow('Editing')
        cv2.setMouseCallback('Editing', callback)
        cv2.imshow('Editing', display_img)

        while True:
            if cv2.waitKey(2)&0xFF==ord('s'):
                print('Saved changes to Alpha')
                cv2.destroyAllWindows()
                # do something on Alpha and Trimap
                Trimap[up: up+T_height, left: left+T_width] = fb_matrix
                Alpha[up: up+T_height, left: left+T_width][fb_matrix==0] = 0
                Alpha[up: up+T_height, left: left+T_width][fb_matrix==2] = 1
                cv2.imshow('out', Trimap*128)
                cv2.imshow('out1', Alpha*255)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                g = maxflow.Graph[float]()
                nodeids = g.add_grid_nodes((T_height, T_width))
                for t_h in range(T_height):  # 0 to T_height-1
                    for t_w in range(T_width):
                        if Trimap[t_h+up, t_w+left] == 1:
                            for n_h, n_w in get_neighbors(t_h, t_w, T_height-1, T_width-1):
                                # Add neighbor edges
                                if Trimap[n_h+up, n_w+left] == 1:
                                    g.add_edge(nodeids[t_h, t_w], nodeids[n_h, n_w], 
                                            get_nlink(t_h+up, t_w+left, n_h+up, n_w+left, src/255, Alpha, 50, beta), 
                                            get_nlink(t_h+up, t_w+left, n_h+up, n_w+left, src/255, Alpha, 50, beta))
                                    # Add s edge and t edge
                                    g.add_tedge(nodeids[t_h, t_w], 
                                            get_stlink(src_cropped[t_h, t_w]/255, GMM_back), # slink
                                            get_stlink(src_cropped[t_h, t_w]/255, GMM_fore))  # tlink
                flow = g.maxflow()
                print(f"Maximum flow: {flow}")
                new_alpha = np.ones((T_height, T_width), np.uint8) * -1
                for t_h in range(T_height):  # 0 to T_height-1
                    for t_w in range(T_width):
                        new_alpha[t_h, t_w] = g.get_segment(nodeids[t_h, t_w])
                new_alpha = 1 - new_alpha
                Alpha[up: up+T_height, left: left+T_width][fb_matrix==1] = \
                        new_alpha[fb_matrix==1]

                break
            elif cv2.waitKey(2)&0xFF==ord('a'):
                print('Abondon the change')
                cv2.destroyAllWindows()
                break
            elif cv2.waitKey(2)&0xFF==ord('b'):
                print('Changing to: background brush')
                isForeBrush = False
            elif cv2.waitKey(2)&0xFF==ord('f'):
                print('Changing to: foreground brush')
                isForeBrush = True
            elif cv2.waitKey(2)&0xFF==ord('q'):
                print('Quit the program')
                cv2.destroyAllWindows()
                exit()
            elif cv2.waitKey(2)&0xFF==ord('w'):
                print('Saving current foreground image')
                

    

img = cv2.imread('imgs/3_src_resized.jpg')
print(img.shape)

left, up, t_width, t_height = cv2.selectROI('roi', img, False, False )
cv2.destroyAllWindows()
GrabCut_kernel(img, up, left, t_height, t_width, 10)


