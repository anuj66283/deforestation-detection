import os
import torch
import shutil
import numpy as np
import torch.nn.functional as F
from shapely.geometry import Polygon

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'deep.pth')
TMP_PATH = os.path.join(os.path.dirname(__file__), '..', 'tmp')
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg'] 

# delete and remake tmp directory
def reset_path():
    shutil.rmtree(TMP_PATH, ignore_errors=True)
    os.mkdir(TMP_PATH)

# returns proper polygon
def to_our(jsn):
    tmp = []
    for x in jsn:
        tmp.append([x['lng'], x['lat']])

    return Polygon(tmp)

# normalize the image
def normalize(arr, a, b):
    one_per = np.percentile(arr, 1)
    nine_per = np.percentile(arr, 99)
    return ((arr-one_per) * ((b-a)/(nine_per-one_per))+a)


# pad image
def pad_image(img, divi=16):
    image = torch.from_numpy(img)

    _, h, w = image.shape

    p_h = ((h + divi - 1) // divi) * divi
    p_w = ((w + divi - 1) // divi) * divi
    
    p_t = (p_h - h) // 2
    p_b = p_h - h - p_t
    p_l = (p_w - w) // 2
    p_r = p_w - w - p_l
    
    image = F.pad(image, (p_l, p_r, p_t, p_b), mode='constant', value=0)
    
    return image

