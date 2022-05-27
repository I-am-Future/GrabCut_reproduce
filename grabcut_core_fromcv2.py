# grabcut_core_fromcv2.py
# My reproduction to the GrubCut with OpenCV-C++'s source code. 

import cv2
import numpy as np
from grabcut_utils_fromcv2 import *
from typing import Tuple
import maxflow

def classify_components(img: np.ndarray, alpha: np.ndarray, 
        bkgmodel: GMM, fkgmodel: GMM) -> np.ndarray:
    ''' The step 1 in the GrabCut Algorithm. '''
    height, width = img.shape[0], img.shape[1]
    result = np.zeros((height, width), dtype=np.uint8)   # k vector
    for h in range(height):
        for w in range(width):
            if alpha[h, w] == 1:
                result[h, w] = fkgmodel.classify(img[h, w])
            else:  # == 0
                result[h, w] = bkgmodel.classify(img[h, w])
    return result


def learn_GMMs(img: np.ndarray, alpha: np.ndarray, k_vec: np.ndarray, 
        bkgmodel: GMM, fkgmodel: GMM) -> None:
    ''' The step 2 in the GrabCut Algorithm. '''
    height, width = img.shape[0], img.shape[1]
    bkgmodel.init_learning()
    fkgmodel.init_learning()
    for h in range(height):
        for w in range(width):
            if alpha[h, w] == 1:  # change on fore GMM
                fkgmodel.add_sample(img[h, w], k_vec[h, w])
            else:  # a == 0
                bkgmodel.add_sample(img[h, w], k_vec[h, w])
    bkgmodel.end_learning()
    fkgmodel.end_learning()


def construct_graph(img: np.ndarray, trimap: np.ndarray, 
        bkgmodel: GMM, fkgmodel: GMM, lambda_: float, 
        gamma: float, beta: float) -> Tuple[np.ndarray, maxflow.GraphFloat]:
    ''' The step 3.a in the GrabCut Algorithm. '''
    height, width = img.shape[0], img.shape[1]
    graph = maxflow.Graph[float]()
    nodeids = graph.add_grid_nodes((height, width))
    for h in range(height):  # 0 to height-1
        for w in range(width):
            # Add s edge and t edge
            if trimap[h, w] == 0:    # must pure background
                graph.add_tedge(nodeids[h, w], 0, lambda_)
            elif trimap[h, w] == 2:  # must pure foreground 
                graph.add_tedge(nodeids[h, w], lambda_, 0)
            else:                    # maybe, who knows
                source_edge = -np.log(bkgmodel.forward_sum(img[h, w]))
                target_edge = -np.log(fkgmodel.forward_sum(img[h, w]))
                graph.add_tedge(nodeids[h, w], source_edge, target_edge)

            for n_h, n_w in get_neighbors(h, w, height-1, width-1):
                # Add neighbor edges
                edge = get_nlink(h, w, n_h, n_w, img, gamma, beta)
                graph.add_edge(nodeids[h, w], nodeids[n_h, n_w], edge, edge)
    return nodeids, graph


def segmentation(graph: maxflow.GraphFloat, nodeids: np.ndarray, 
        height: int, width: int) -> np.ndarray:
    ''' The step 3.b in the GrabCut Algorithm. '''
    flow = graph.maxflow()
    print(f"Maximum flow: {flow}")
    new_alpha = np.zeros((height, width), np.uint8)
    for h in range(height):  # 0 to T_height-1
        for w in range(width):
            new_alpha[h, w] = graph.get_segment(nodeids[h, w])
    new_alpha = 1 - new_alpha
    # np.save(f'saved_npy/save{epoch+1}.npy', new_alpha)
    return new_alpha
    

def GrabCut_kernel(
    src: np.ndarray,  # the input image
    up: int, 
    left: int, 
    T_height: int, 
    T_width: int,
    n_epoches: int, 
    interact: bool = False
) -> np.ndarray:
    ''' The core of the GrabCut Algorithm. '''

    ## initialization
    print('GrabCut Core started. img size:', src.shape)
    print(f'ROI position: rectangle {up} to top, {left} to left, with size {T_height} * {T_width}' )
    height, width = src.shape[0], src.shape[1]
    src = src.astype(np.float32)
    # Trimap: 0: Background, 1: Undefined, 2: Foreground
    Trimap = np.zeros((height, width), dtype=np.uint8) 
    Trimap[up: up+T_height, left: left+T_width] = 1
    # Alpha: 0: Background, 1: Foreground
    Alpha = Trimap.copy()
    # args: 
    gamma, lambda_, beta = 50, 450, calc_beta(src)
    print(f'Args: beta={beta}, gamma={gamma}, lambda={lambda_}')

    # Init GMMs
    GMM_back = GMM()
    back_pixels = select_pixels(src, Alpha, 0)
    GMM_back.init_param(back_pixels)
    GMM_fore = GMM()
    fore_pixels = select_pixels(src, Alpha, 1)
    GMM_fore.init_param(fore_pixels)    
    print('Initialization completed!')

    for epoch in range(n_epoches):
        ## step 1: Assign GMM components to pixels
        k_vec = classify_components(src, Alpha, GMM_back, GMM_fore)  # k vector
        print(f'epoch {epoch+1} :: step 1 finished!')

        ## step 2: Learn GMM parameters
        learn_GMMs(src, Alpha, k_vec, GMM_back, GMM_fore)
        print(f'epoch {epoch+1} :: step 2 finished!')

        ## step 3: Estimate segmentation
        nodeids, g = construct_graph(src, Trimap, GMM_back, GMM_fore, lambda_, gamma, beta)

        Alpha = segmentation(g, nodeids, height, width)
        print(f'epoch {epoch+1} :: step 3 finished!')

        print(f'epoch {epoch+1} of {n_epoches}')

    if not interact:
        return Alpha

    ### Further Interaction

    def prompt():
        ''' Prompt in the UI. '''
        print('s: save change and recalculate')
        print('a: abandon the change')
        print('b: change brush to background')
        print('f: change brush to foreground')
        print('q: quit the function')
        print('w: write the image to \'Output_img.jpg\'')
        print('Changing to: foreground brush \n')

    def callback(event, x, y, flags, param):
        ''' Callback function of the window. '''
        nonlocal isForeBrush
        if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON and \
                0 <= x <= width and 0 <= y <= height:
            if isForeBrush:
                cv2.circle(fb_matrix, (x, y), 2, (2, ), -1)
                cv2.circle(display_img, (x, y), 2, (255, 255, 255), -1)
            else:
                cv2.circle(fb_matrix, (x, y), 2, (0, ), -1)
                cv2.circle(display_img, (x, y), 2, (0, 0, 255), -1)
            cv2.imshow('Editing', np.hstack((display_img, masked_img)))

    working = True
    while working:  # every whole editing iteration
        prompt()
        fb_matrix = Trimap.copy()
        isForeBrush = True
        masked_img = cv2.bitwise_or(src, np.zeros_like(src), mask=Alpha).astype(np.uint8)
        display_img = src.copy().astype(np.uint8)   # for real-time visualization
        cv2.namedWindow('Editing')
        cv2.setMouseCallback('Editing', callback)
        cv2.imshow('Editing', np.hstack((display_img, masked_img)))
    
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                print('Saved changes and re-calculate...')
                cv2.destroyAllWindows()
                # do something on Alpha and Trimap
                Trimap = fb_matrix
                Alpha[fb_matrix==2] = 1
                Alpha[fb_matrix==0] = 0

                k_vec = classify_components(src, Alpha, GMM_back, GMM_fore)  # k vector

                ## step 2: Learn GMM parameters
                learn_GMMs(src, Alpha, k_vec, GMM_back, GMM_fore)

                nodeids, g = construct_graph(src, Trimap, 
                    GMM_back, GMM_fore, lambda_, gamma, beta)
                Alpha = segmentation(g, nodeids, height, width)
                break
            elif key == ord('a'):
                print('Abondon the change')
                cv2.destroyAllWindows()
                break

            elif key == ord('b'):
                print('Changing to: background brush')
                isForeBrush = False

            elif key == ord('f'):
                print('Changing to: foreground brush')
                isForeBrush = True

            elif key == ord('q'):
                print('Quit the ui program')
                cv2.destroyAllWindows()
                working = False
                break                

            elif key == ord('w'):
                print('Saving current image')
                cv2.imwrite('Output_img.jpg', masked_img)
                working = False
                break  
    return Alpha              
                


