import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import random
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

consider_cc_thresh = 10
split_cc_thresh = 1500


def tidy_cc(x, y, w_size=41):
    # Opens image
    imgRGB = cv2.imread(x)
    img = rgb2gray(imgRGB)
    img = img_as_uint(img)
    label = cv2.imread(y, 0)

    features = []
    labels = []
    # Binarizes image
    global_thresh = threshold_otsu(img)
    binary_global = img < global_thresh

    # Calculates connected components
    label_image = measure.label(binary_global, neighbors=8, background=0)

    properties = regionprops(label_image, intensity_image=binary_global)

    # Iterates through connected components
    for p in properties:
        if p['area'] > consider_cc_thresh:
            # Gets the position of the connected component
            slices = p['slice']

            # Calculates width and height of the connected component
            h = slices[0].stop-slices[0].start
            w = slices[1].stop-slices[1].start

            # Calculates rescale rate
            rescale_rate = max(h/9, w/9)

            # Calculates the height and width of the cut in the original
            # image
            height_rescaled = rescale_rate*w_size
            width_rescaled = rescale_rate*w_size

            # Calculates the pad around the CC window
            pad_height = height_rescaled - h
            pad_width = width_rescaled - w

            # Counts the most frequent label over the connected component
            mask = p['image'].flat
            accountant_indexes = np.where(mask == True)
            counts = np.bincount(label[slices].flat[accountant_indexes[0]])
            label_value = np.argmax(counts)

            if label_value == 1:
                labels.append(label_value)
            else:
                labels.append(0)

            # Calculates the coordinates of the window around the CC
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

            # If the window is outside the image, fills it with zero
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

            # Resizes window to a standard dimension
            normalized = cv2.resize(neighbourhood, dsize=(
                w_size, w_size), interpolation=cv2.INTER_CUBIC)
            features.append(normalized)

    return np.array(features), np.array(labels)


def tidy_pixels(x, y, w_size=41):
    # Opens image
    imgRGB = cv2.imread(x)
    label = cv2.imread(y, 0)
    img = rgb2gray(imgRGB)
    img = img_as_uint(img)

    # Binarizes image
    global_thresh = threshold_otsu(img)
    binary_global = img < global_thresh

    # Calculates the amount of random pixels that will be picked
    pixels_number = int(np.rint((imgRGB.shape[0]*imgRGB.shape[1])*0.0004))

    features = np.zeros(
        (pixels_number, w_size, w_size, 3), dtype=np.uint8)
    labels = np.zeros(pixels_number, dtype=np.uint8)

    # Iterates through random pixels
    i = 0
    while(i < pixels_number):
        x = random.randrange(0, np.shape(imgRGB)[0], 1)
        y = random.randrange(0, np.shape(imgRGB)[1], 1)

        # Considers only pixels of the foreground
        if binary_global[x][y] == True:
            label_value = label[x][y]
            if label_value == 1:
                labels[i] = 1
            else:
                labels[i] = 0

            # Calculates the coordinates of the window around the CC
            top_fill = 0
            bottom_fill = 0
            left_fill = 0
            right_fill = 0

            if x - int(w_size/2) >= 0:
                x1 = x - int(w_size/2)
            else:
                top_fill = abs(x - int(w_size/2))
                x1 = 0
            if x + int(w_size/2) + 1 < imgRGB.shape[0]:
                y1 = x + int(w_size/2) + 1
            else:
                bottom_fill = abs(imgRGB.shape[0] - (x + int(w_size/2)+1))
                y1 = imgRGB.shape[0]

            if y - (int(w_size/2) + 1) >= 0:
                x2 = y - int(w_size/2)
            else:
                left_fill = abs(y - int(w_size/2))
                x2 = 0
            if y + int(w_size/2)+1 < imgRGB.shape[1]:
                y2 = y + int(w_size/2)+1
            else:
                right_fill = abs(imgRGB.shape[1] - (y + int(w_size/2)+1))
                y2 = imgRGB.shape[1]

            window = (slice(x1, y1), slice(x2, y2))
            neighbourhood = imgRGB[window]

            # If the window is outside the image, fills it with zero
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
            features[i] = neighbourhood
        else:
            i -= 1
        i += 1

    return features, labels


def tidy_superpixels(x, y, w_size=41):
    # Opens image
    imgRGB = cv2.imread(x)
    img = rgb2gray(imgRGB)
    img = img_as_uint(img)
    label = cv2.imread(y, 0)

    features = []
    labels = []

    # Binarizes image
    global_thresh = threshold_otsu(img)
    binary_global = img < global_thresh

    # Calculates connected components
    label_image, n = measure.label(
        binary_global, neighbors=8, background=0, return_num=True)

    seg_prop = 0.011

    properties = regionprops(label_image, intensity_image=binary_global)

    # Iterates through connected components
    for region in properties:
        if region['convex_area'] > split_cc_thresh:
            segments = math.ceil(region['convex_area'] * seg_prop)

            slices = region['slice']

            img_x = slices[0].start
            img_y = slices[1].start

            # Splits big CCs into superpixels
            segments_slic = slic(image=imgRGB[slices], n_segments=segments)

            for i in range(segments_slic.shape[0]):
                for j in range(segments_slic.shape[1]):
                    if(region['image'][i][j]):
                        label_image[img_x + i][img_y +
                                               j] = int(segments_slic[i][j]) + n
            n += segments

    properties = regionprops(label_image, intensity_image=binary_global)

    # Iterates through CCs and superpixels
    for p in properties:
        if p['area'] > consider_cc_thresh:
            # Gets the position of the region
            slices = p['slice']

            # Calculates width and height of the connected component
            h = slices[0].stop-slices[0].start
            w = slices[1].stop-slices[1].start

            # Calculates rescale rate
            rescale_rate = max(h/9, w/9)

            # Calculates the height and width of the cut in the original
            # image
            height_rescaled = rescale_rate*w_size
            width_rescaled = rescale_rate*w_size

            # Calculates the pad around the region window
            pad_height = height_rescaled - h
            pad_width = width_rescaled - w

            # Counts the most frequent label over the region
            mask = p['image'].flat
            accountant_indexes = np.where(mask == True)
            counts = np.bincount(label[slices].flat[accountant_indexes[0]])
            label_value = np.argmax(counts)

            if label_value == 1:
                labels.append(label_value)
            else:
                labels.append(0)

            # Calculates the coordinates of the window around the region
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

            # If the window is outside the image, fills it with zero
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

            # Resizes window to a standard dimension
            normalized = cv2.resize(neighbourhood, dsize=(
                w_size, w_size), interpolation=cv2.INTER_CUBIC)
            features.append(normalized)

    return np.array(features), np.array(labels)