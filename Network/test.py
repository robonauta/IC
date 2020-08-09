import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import math

from skimage import measure
from skimage.color import rgb2gray, label2rgb
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage.util import img_as_float, img_as_uint, img_as_float32, invert
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.segmentation import slic

import cv2

label = cv2.imread('00000357_label.png', 0)

cv2.imwrite('gt.png', label*255)
'''plt.imshow(label*255, cmap='gray')
plt.show()'''
