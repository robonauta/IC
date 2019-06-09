import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import math

from skimage import measure
from skimage.color import rgb2gray, label2rgb
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage.util import img_as_float, img_as_uint, img_as_float32
from skimage.io import imread, imshow
from skimage.transform import resize

import cv2


def tidy_cc(x, y):
    imgRGB = cv2.imread(x)
    img = rgb2gray(imgRGB)
    img = img_as_uint(img)
    label = cv2.imread(y, 0)

    features = []
    labels = []

    global_thresh = threshold_otsu(img)
    binary_global = img < global_thresh

    label_image = measure.label(binary_global, neighbors=8, background=0)

    properties = regionprops(label_image, intensity_image=binary_global)

    # DEBUG
    '''
    i = 0
    '''
    for p in properties:
        if p['area'] > 10:
            slices = p['slice']
            h = slices[0].stop-slices[0].start
            w = slices[1].stop-slices[1].start

            rescale_rate = max(h/9, w/9)

            height_rescaled = rescale_rate*41
            width_rescaled = rescale_rate*41

            pad_height = height_rescaled - h
            pad_width = width_rescaled - w

            counts = np.bincount(label[slices].flat)
            label_value = np.argmax(counts)

            if label_value == 1:
                labels.append(label_value)
            else:
                labels.append(0)

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

            # DEBUG
            '''
            connectedComponent = img[p['slice']]
            '''

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

            # DEBUG
            '''
            plt.imshow(connectedComponent, cmap='gray')
            plt.title('Connected component')
            plt.show()
            plt.savefig('imgs/cc/cc'+str(i))
            plt.clf()

            plt.imshow(neighbourhood, cmap='gray')
            plt.title('Neighbourhood')
            plt.show()
            plt.savefig('imgs/nh/nh'+str(i))
            plt.clf()
            '''

            normalized = cv2.resize(neighbourhood, dsize=(
                41, 41), interpolation=cv2.INTER_CUBIC)
            features.append(normalized)

            # DEBUG
            '''
            plt.imshow(normalized, cmap = 'gray')
            plt.title('Normalized')
            plt.gca().add_patch(Rectangle((15, 15), 9, 9, linewidth=1,
                                        edgecolor='r', facecolor='none'))
            plt.show()
            plt.savefig('imgs/normal/normal'+str(i))
            plt.clf()
            i += 1
            '''
    return np.array(features), np.array(labels)


def tidy_pixels(x, y):
    imgRGB = cv2.imread(x)
    label = cv2.imread(y, 0)

    features = []
    labels = []

    for x in range(np.shape(imgRGB)[0]):
        for y in range(np.shape(imgRGB)[1]):
            label_value = label[x][y]
            if label_value == 1:
                labels.append(label_value)
            else:
                labels.append(0)
            top_fill = 0
            bottom_fill = 0
            left_fill = 0
            right_fill = 0

            if x - 20 >= 0:
                x1 = x - 20
            else:
                top_fill = abs(x - 20)
                x1 = 0
            if x + 21 < imgRGB.shape[0]:
                y1 = x + 21
            else:
                bottom_fill = abs(imgRGB.shape[0] - (x + 21))
                y1 = imgRGB.shape[0]

            if y - 20 >= 0:
                x2 = y - 20
            else:
                left_fill = abs(y - 20)
                x2 = 0
            if y + 21 < imgRGB.shape[1]:
                y2 = y + 21
            else:
                right_fill = abs(imgRGB.shape[1] - (y + 21))
                y2 = imgRGB.shape[1]

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

            # DEBUG
            '''
            print(neighbourhood.shape)
            plt.imshow(neighbourhood)
            plt.title('Neighbourhood')
            plt.show()
            #plt.savefig('imgs/nh/nh'+str(i))
            plt.clf()
            '''
            features.append(neighbourhood)
    return np.array(features), np.array(labels)


def main():
    f, l = tidy_pixels('00000085.jpg', '00000085_label.png')
    print(f.shape)


if __name__ == "__main__":
    main()
