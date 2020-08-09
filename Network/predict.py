import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import math

from conv import conv_cc, conv_pixels, conv_superpixels
from skimage import measure
from skimage.color import rgb2gray, label2rgb
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage.util import img_as_float, img_as_uint, img_as_float32, invert
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.segmentation import slic

import cv2

from tensorflow.keras.models import load_model


def predict_on_cc():
    model = conv_cc()

    imgRGB = cv2.imread('00000357.jpg')
    img = rgb2gray(imgRGB)
    img = img_as_uint(img)

    global_thresh = threshold_otsu(img)
    binary_global = img < global_thresh

    label_image = measure.label(binary_global, neighbors=8, background=0)

    properties = regionprops(label_image, intensity_image=binary_global)

    result = np.zeros(imgRGB.shape, dtype=np.uint8)

    for p in properties:
        slices = p['slice']
        h = slices[0].stop-slices[0].start
        w = slices[1].stop-slices[1].start

        rescale_rate = max(h/9, w/9)

        height_rescaled = rescale_rate*41
        width_rescaled = rescale_rate*41

        pad_height = height_rescaled - h
        pad_width = width_rescaled - w

        top_fill = 0
        bottom_fill = 0
        left_fill = 0
        right_fill = 0

        if slices[0].start - math.floor(pad_height/2.0) >= 0:
            x1 = slices[0].start - math.floor(pad_height/2)
        else:
            top_fill = abs(slices[0].start - math.floor(pad_height/2))
            x1 = 0
        if slices[0].stop + math.ceil(pad_height/2.0) < img.shape[0]:
            y1 = slices[0].stop + math.ceil(pad_height/2.0)
        else:
            bottom_fill = abs(
                img.shape[0] - (slices[0].stop + math.ceil(pad_height/2.0)))
            y1 = img.shape[0] - 1

        if slices[1].start - math.floor(pad_width/2.0) >= 0:
            x2 = slices[1].start - math.floor(pad_width/2)
        else:
            left_fill = abs(
                slices[1].start - math.floor(pad_width/2.0))
            x2 = 0
        if slices[1].stop + math.ceil(pad_width/2.0) < img.shape[1]:
            y2 = slices[1].stop + math.ceil(pad_width/2.0)
        else:
            right_fill = abs(
                img.shape[1] - (slices[1].stop + math.ceil(pad_width/2.0)))
            y2 = img.shape[1] - 1

        window = (slice(x1, y1), slice(x2, y2))

        neighbourhood = imgRGB[window]

        if top_fill != 0:
            neighbourhood = np.append(
                np.zeros((top_fill, neighbourhood.shape[1], 3), dtype=np.uint8), neighbourhood, axis=0)
        if bottom_fill != 0:
            neighbourhood = np.append(neighbourhood, np.zeros(
                (bottom_fill, neighbourhood.shape[1], 3), dtype=np.uint8), axis=0)
        if left_fill != 0:
            neighbourhood = np.append(
                np.zeros((neighbourhood.shape[0], left_fill, 3), dtype=np.uint8), neighbourhood, axis=1)
        if right_fill != 0:
            neighbourhood = np.append(neighbourhood, np.zeros(
                (neighbourhood.shape[0], right_fill, 3), dtype=np.uint8), axis=1)

        normalized = cv2.resize(neighbourhood, dsize=(
            41, 41), interpolation=cv2.INTER_CUBIC)

        l = model.predict(normalized.reshape(
            (1, normalized.shape[0], normalized.shape[1], normalized.shape[2])))
        cc_index = np.where(p['image'])
        result[p['slice']][cc_index] = np.rint(np.squeeze(l)).astype('uint8')

    plt.imshow(result*255)
    plt.show()


def main():
    predict_on_cc()


if __name__ == "__main__":
    main()
