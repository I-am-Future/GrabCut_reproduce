import cv2
import numpy as np
from grabcut_utils_improved import *
import maxflow

from grabcut_utils_improved import calc_beta

def GrabCut_kernel(
    src: np.ndarray,  # the input image
    up: int, 
    left: int, 
    T_height: int, 
    T_width: int,
    n_epoches: int, 
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
    beta = calc_beta(src)
    print(beta)
    src = src.astype(np.float32)
    src_cropped = src[up: up+T_height, left: left+T_width].copy()
    # Init GMMs
    GMM_back = GMM()
    back_pixels = select_pixels(src, Alpha, 0)
    GMM_back.init_param(back_pixels)
    GMM_fore = GMM()
    fore_pixels = select_pixels(src, Alpha, 1)
    GMM_fore.init_param(fore_pixels)    
    # import pickle
    # pickle.dump(GMM_back, open('gmmb.bin', 'wb'))
    # pickle.dump(GMM_fore, open('gmmf.bin', 'wb'))
    # GMM_back = pickle.load(open('gmmb.bin', 'rb'))
    # GMM_fore = pickle.load(open('gmmf.bin', 'rb'))
    print('Init learning completed!')

    for epoch in range(n_epoches):
        ## step 1: Assign GMM components to pixels
        k_vec = np.zeros_like(Trimap)  # k vector
        for h in range(height):
            for w in range(width):
                # if Trimap[h, w] == 1:  # in T_u
                    if Alpha[h, w] == 1:
                        k_vec[h, w] = GMM_fore.classify(src[h, w])
                    else:  # == 0
                        k_vec[h, w] = GMM_back.classify(src[h, w])
        print(f'epoch {epoch+1} :: step 1 finished!')
        ## step 2: Learn GMM parameters
        GMM_back.init_learning()
        GMM_fore.init_learning()
        for h in range(height):
            for w in range(width):
                if Alpha[h, w] == 1:  # change on fore GMM
                    GMM_fore.add_sample(src[h, w], k_vec[h, w])
                else:  # a == 0
                    GMM_back.add_sample(src[h, w], k_vec[h, w])
        GMM_back.end_learning()
        GMM_fore.end_learning()
        print(f'epoch {epoch+1} :: step 2 finished!')
        ## step 3: Estimate segmentation
        gamma = 50
        lambda_ = 450
        g = maxflow.Graph[float]()
        nodeids = g.add_grid_nodes((height, width))
        for h in range(height):  # 0 to height-1
            for w in range(width):
                # Add s edge and t edge
                if Trimap[h, w] == 0:    # must pure background
                    source_edge = 0
                    target_edge = lambda_
                elif Trimap[h, w] == 2:  # must pure foreground
                    source_edge = lambda_
                    target_edge = 0   
                else:                    # maybe, who knows
                    source_edge = -np.log(GMM_back.forward_sum(src[h, w]))
                    target_edge = -np.log(GMM_fore.forward_sum(src[h, w]))
                g.add_tedge(nodeids[h, w], source_edge, target_edge)

                for n_h, n_w in get_neighbors(h, w, height-1, width-1):
                    # Add neighbor edges
                    g.add_edge(nodeids[h, w], nodeids[n_h, n_w], 
                            get_nlink(h, w, n_h, n_w, src, gamma, beta), 
                            get_nlink(h, w, n_h, n_w, src, gamma, beta))
        print('Graph constructed!')
        flow = g.maxflow()
        print(f"Maximum flow: {flow}")
        new_alpha = np.zeros((height, width), np.uint8)
        for h in range(height):  # 0 to T_height-1
            for w in range(width):
                new_alpha[h, w] = g.get_segment(nodeids[h, w])
        new_alpha = 1 - new_alpha
        # print(new_alpha)  # 0: back, 1: fore
        np.save(f'saved_npy/save{epoch+1}.npy', new_alpha)
        Alpha = new_alpha
        print(f'epoch {epoch+1} of {n_epoches}')
    exit() 

    ## User Editting
    src_cropped = src_cropped.astype(np.uint8)
    Alpha = Alpha.astype(np.uint8)
    def callback(event, x, y, flags, param):
        ''' Callback function of the window. '''
        nonlocal isForeBrush
        if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            if isForeBrush:
                cv2.circle(fb_matrix, (x, y), 3, (2, ), -1)
                cv2.circle(display_img, (x, y), 3, (255, 255, 255), -1)
            else:
                cv2.circle(fb_matrix, (x, y), 3, (0, ), -1)
                cv2.circle(display_img, (x, y), 3, (0, 0, 255), -1)
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
            if cv2.waitKey(1)&0xFF==ord('s'):
                print('Saved changes to Alpha')
                cv2.destroyAllWindows()
                # do something on Alpha and Trimap
                diff = Trimap[up: up+T_height, left: left+T_width] != fb_matrix
                Trimap[up: up+T_height, left: left+T_width][diff] = fb_matrix[diff]
                Alpha[up: up+T_height, left: left+T_width][fb_matrix==0] = 0
                Alpha[up: up+T_height, left: left+T_width][fb_matrix==2] = 1
                cv2.imshow('out', Trimap*128)
                cv2.imshow('out1', Alpha*255)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                print('step 1 revisit')
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
                print('step 2 revisit')
                for a in range(2):
                    for k in range(5):
                        pixels = src[ (Alpha==a) & (k_vec==k) ] / 255  # shape: m * 3
                        if len(pixels) == 0:
                            continue
                        if a == 0:  # change on back GMM
                            GMM_back.mu[k] = np.mean(pixels, axis=0)
                            GMM_back.sigma[k] = np.cov(pixels, rowvar=False)
                            GMM_back.mixCoef[k] = len(pixels) / np.sum(Alpha==a)
                        else:  # a == 1
                            GMM_fore.mu[k] = np.mean(pixels, axis=0)
                            GMM_fore.sigma[k] = np.cov(pixels, rowvar=False)
                            GMM_fore.mixCoef[k] = len(pixels) / np.sum(Alpha==a)
                GMM_back.update_()
                GMM_fore.update_()
                print('step 3 revisit')
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
            elif cv2.waitKey(1)&0xFF==ord('a'):
                print('Abondon the change')
                cv2.destroyAllWindows()
                break
            elif cv2.waitKey(1)&0xFF==ord('b'):
                print('Changing to: background brush')
                isForeBrush = False
            elif cv2.waitKey(1)&0xFF==ord('f'):
                print('Changing to: foreground brush')
                isForeBrush = True
            elif cv2.waitKey(1)&0xFF==ord('q'):
                print('Quit the program')
                cv2.destroyAllWindows()
                exit()
            elif cv2.waitKey(1)&0xFF==ord('w'):
                np.save('saved_npy/1_mask.npy', Alpha)
                print('Saving current foreground image')
                

    

img = cv2.imread('imgs/1_src_resized.jpg')
print(img.shape)

# left, up, t_width, t_height = cv2.selectROI('roi', img, False, False )
# cv2.destroyAllWindows()
up, left, t_height, t_width = 12, 11, 180, 265
GrabCut_kernel(img, up, left, t_height, t_width, 50)


