import matplotlib.pyplot as plt
import cv2
import numpy as np
import random


def plot_img_and_mask(img, mask, index,epoch,save_dir):
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    # plt.show()
    plt.savefig(save_dir+"/batch_{}_{}_seg.png".format(epoch,index))

def show_seg_result(img, result, index, epoch, save_dir=None, is_ll=False,palette=None,is_demo=False,is_gt=False):
    # img = mmcv.imread(img)
    # img = img.copy()
    # seg = result[0]
    if palette is None:
        palette = np.random.randint(
                0, 255, size=(3, 3))
    palette[0] = [0, 0, 0]
    palette[1] = [0, 255, 0]
    palette[2] = [255, 0, 0]
    palette = np.array(palette)
    assert palette.shape[0] == 3 # len(classes)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    
    if not is_demo:
        color_seg = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[result == label, :] = color
    else:
        color_area = np.zeros((result[1].shape[0], result[1].shape[1], 3), dtype=np.uint8)
        
        # for label, color in enumerate(palette):
        #     color_area[result[0] == label, :] = color

        # color_area[result[0] == 1] = [0, 255, 0]
        color_area[result[1] == 255] = [0, 255, 0]
        color_seg = color_area

    # convert to BGR
    color_seg = color_seg[..., ::-1]
    # print(color_seg.shape)
    color_mask = np.mean(color_seg, 2)
    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    # img = img * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    # img = cv2.resize(img, (1280,720), interpolation=cv2.INTER_LINEAR)

    if not is_demo:
        if not is_gt:
            if not is_ll:
                cv2.imwrite(save_dir+"/batch_{}_{}_da_segresult.png".format(epoch,index), img)
            else:
                cv2.imwrite(save_dir+"/batch_{}_{}_ll_segresult.png".format(epoch,index), img)
        else:
            if not is_ll:
                cv2.imwrite(save_dir+"/batch_{}_{}_da_seg_gt.png".format(epoch,index), img)
            else:
                cv2.imwrite(save_dir+"/batch_{}_{}_ll_seg_gt.png".format(epoch,index), img)  
    return img

    

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    
    tl = line_thickness or round(0.0001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or (255,200,0)
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    '''
    _______________
    |x0,x1        |  
    |        x2,x3|
    ---------------

    '''
    # Start coordinate represents the top left corner of rectangle 
    # Ending coordinate represents the bottom right corner of rectangle

    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)