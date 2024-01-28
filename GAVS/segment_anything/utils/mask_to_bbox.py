import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from skimage.measure import label, regionprops, find_contours
import torch
import matplotlib.pyplot as plt

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" Convert a mask to border image """
def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 1)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border

def filter_bboxes(bboxes):
    valid = []
    for i, bbox in enumerate(bboxes):
        is_contained = False
        for j, other_bbox in enumerate(bboxes):
            # 如果当前bounding-box和其他bounding-box是同一个bounding-box，则跳过
            if i == j:  # 如果当前bounding-box被其他bounding-box包含，则标记为被包含并退出循环
                continue
            if bbox[0] >= other_bbox[0] and bbox[1] >= other_bbox[1] and bbox[2] <= other_bbox[2] and bbox[3] <= other_bbox[3]:
                is_contained = True
                break
        if not is_contained:
            valid.append(bbox)
    return valid

""" Mask to bounding boxes """
def mask_to_bbox(mask):
    """ From {mask} to {bboxes}, in Numpy format.  """
    bboxes = []
    mask = mask_to_border(mask)
    # print(np.sum(mask))
    lbl = label(mask)
    props = regionprops(lbl)
    
    # (min_row, min_col, max_row, max_col)
    for prop in props:
        x1 = prop.bbox[1]  # x_min
        y1 = prop.bbox[0]  # y_min 

        x2 = prop.bbox[3]   
        y2 = prop.bbox[2]  
        
        if (x2-x1) * (y2-y1) <= 1000:
            continue
        bboxes.append([x1, y1, x2, y2])
    print("no-ck", bboxes)

    valid_bboxes = filter_bboxes(bboxes)
    if len(valid_bboxes) == 0:
        valid_bboxes = [[1, 1, 2, 2]]  # FIXME: 可以提升
        input()
        print(valid_bboxes)
        input()
    return valid_bboxes

def parse_mask(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask



def from_mask_to_bboxes(mask):
    """ Read mask from {cv2.IMREAD_COLOR} to {cv2.COLOR_BGR2GRAY} and return {bboxes}. """
    H, W = mask.shape[0], mask.shape[1]
    
    # print("[1]", np.min(mask), np.max(mask), mask.shape)  # [343, 610]  # TODO: 如何让二者相同...
    # print("[2]", np.min(y), np.max(y), y.shape)           # [720, 1280]
    
    mask = 255 - mask  # 取反
    
    """ Detecting bounding boxes """
    bboxes = mask_to_bbox(mask)
    print("[OG] bboxes:", bboxes)
    
    """ marking bounding box on image """
    for bbox in bboxes:
        x = cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        # print(">>>", x.shape)
    
    def _nf(_bi, _v): return _bi / _v
    normed_bboxes = []
    for pbb in bboxes:
        pbb[0], pbb[2], pbb[1], pbb[3] = _nf(pbb[0], W), _nf(pbb[2], W), _nf(pbb[1], H), _nf(pbb[3], H)
        pred_bbox = [pbb[0], pbb[1], pbb[2], pbb[3]]
        normed_bboxes.append(pred_bbox)
    # print('norm:', normed_bboxes)
    """ Saving the image """
    # cat_image = np.concatenate([x], axis=1)
    
    # __p = f"/home/yaoting_wang/segment-anything-main/segment_anything/utils/results/board.png"
    # if len(bboxes) > 0:
    #     cv2.imwrite(__p, x)
    #     # cv2.imwrite(__p, mask)
    #     print("in-shape", x.shape)
    #     draw_x(__p)
    #     input()
    # else:
    #     print('none-box')
    return bboxes
    ...

def from_mask_path_to_bboxes(mask_path: str, img_path) -> list:
    """ Read mask from {cv2.IMREAD_COLOR} to {cv2.COLOR_BGR2GRAY} and return {bboxes}. """
    y1 = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    y = cv2.cvtColor(y1, cv2.COLOR_BGR2GRAY)
    H, W = y.shape[0], y.shape[1]
    
    print("y_1", y1.shape)
    
    """ Detecting bounding boxes """
    bboxes = mask_to_bbox(y)
    print("[OG] bboxes:", bboxes)
    
    """ marking bounding box on image """
    for bbox in bboxes:
        x = cv2.rectangle(y1, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        print(">>>", x.shape)
    
    def _nf(_bi, _v): return _bi / _v
    normed_bboxes = []
    for pbb in bboxes:
        pbb[0], pbb[2], pbb[1], pbb[3] = _nf(pbb[0], W), _nf(pbb[2], W), _nf(pbb[1], H), _nf(pbb[3], H)
        pred_bbox = [pbb[0], pbb[1], pbb[2], pbb[3]]
        normed_bboxes.append(pred_bbox)
    # print('norm:', normed_bboxes)
    """ Saving the image """
    # cat_image = np.concatenate([x], axis=1)
    
    __p = f"/home/yaoting_wang/segment-anything-main/segment_anything/utils/results/board.png"
    if len(bboxes) > 0:
        cv2.imwrite(__p, x)
        print("in-shape", x.shape)
        draw_x(__p)
    else:
        print('none-box')
    return bboxes

def draw_fig(mask):
    __p = f"/home/yaoting_wang/segment-anything-main/segment_anything/utils/results/board.png"
    cv2.imwrite(__p, mask)

def draw_x(__p):
    # Load image
    img = cv2.imread(__p)

    # Convert image to RGB color space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create figure and axis objects
    fig, ax = plt.subplots()

    # Show image
    ax.imshow(img)

    # # Draw coordinate axis
    # ax.axhline(y=img.shape[0]//2, color='r', linestyle='-', linewidth=1) # x-axis
    # ax.axvline(x=img.shape[1]//2, color='r', linestyle='-', linewidth=1) # y-axis

    # Set x and y tick labels
    x_ticks = list(range(0, img.shape[1], 50))
    y_ticks = list(range(0, img.shape[0], 50))
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([str(x) for x in x_ticks])
    ax.set_yticklabels([str(y) for y in y_ticks])

    # Show plot
    plt.show()
    plt.savefig('/home/yaoting_wang/segment-anything-main/segment_anything/utils/results/b1.png')

if __name__ == "__main__":
    x_s = ['/home/data/AVS/v1s/__GOGlHL23s_4000_9000/frames/0.jpg', '/home/data/AVS/v1s/__GOGlHL23s_4000_9000/frames/1.jpg']
    y_s = ['/home/data/AVS/v1s/__GOGlHL23s_4000_9000/labels_rgb/0.png', '/home/data/AVS/v1s/__GOGlHL23s_4000_9000/labels_rgb/1.png']
    
    
    # """ Load the dataset """
    # images = sorted(glob(os.path.join("data", "image", "*")))
    # masks = sorted(glob(os.path.join("data", "mask", "*")))

    """ Create folder to save images """
    create_dir("results")

    """ Loop over the dataset """
    # for x, y in tqdm(zip(images, masks), total=len(images)):
    for x, y in tqdm(zip(x_s, y_s), total=len(x_s)):
        """ Extract the name """
        name = x.split("/")[-1].split(".")[0]

        """ Read image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)
        y = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)
        # print(np.sum(y))  # 27431808
        # print(y.shape)  # (720, 1280)

        """ Detecting bounding boxes """
        bboxes = mask_to_bbox(y)
        print(bboxes)
        # input('---')

        """ marking bounding box on image """
        for bbox in bboxes:
            x = cv2.rectangle(x, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

        """ Saving the image """
        cat_image = np.concatenate([x, parse_mask(y)], axis=1)
        cv2.imwrite(f"results/{name}.png", cat_image)