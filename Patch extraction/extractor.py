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

from localslic import SLIC

area_mr_thresh = 10
area_cc = 500

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
        if p['area'] > area_mr_thresh:
            slices = p['slice']
            h = slices[0].stop-slices[0].start
            w = slices[1].stop-slices[1].start

            rescale_rate = max(h/9, w/9)

            height_rescaled = rescale_rate*41
            width_rescaled = rescale_rate*41

            pad_height = height_rescaled - h
            pad_width = width_rescaled - w

            mask = p['image'].flat
            accountant_indexes = np.where(mask == True)
            counts = np.bincount(label[slices].flat[accountant_indexes[0]])
            label_value = np.argmax(counts)

            if label_value == 1:
                labels.append(label_value)
            else:
                labels.append(0)
            # print(labels[-1])
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


def tidy_pixels(x, y, w_size=41):
    imgRGB = cv2.imread(x)
    label = cv2.imread(y, 0)

    features = np.zeros(
        (imgRGB.shape[0]*imgRGB.shape[1], w_size, w_size, 3), dtype=np.uint8)
    labels = np.zeros((imgRGB.shape[0]*imgRGB.shape[1]), dtype=np.uint8)

    i = 0
    for x in range(np.shape(imgRGB)[0]):
        for y in range(np.shape(imgRGB)[1]):
            label_value = label[x][y]
            if label_value == 1:
                labels[i] = 1
            else:
                labels[i] = 0
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
            # print('t: '+str(top_fill)+' b: '+str(bottom_fill) +
            #      ' l: '+str(left_fill)+' r: '+str(right_fill))
            # print(window)

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
            # plt.savefig('imgs/nh/nh'+str(i))
            plt.clf()
            '''
            features[i] = neighbourhood
            i += 1
    return features, labels


def tidy_superpixels(x, y):
    imgRGB = cv2.imread(x)
    img = rgb2gray(imgRGB)
    img = img_as_uint(img)
    label = cv2.imread(y, 0)

    features = []
    labels = []

    global_thresh = threshold_otsu(img)
    binary_global = img < global_thresh

    label_image, n = measure.label(
        binary_global, neighbors=8, background=0, return_num=True)

    area_thresh = 100
    seg_prop = 0.011

    properties = regionprops(label_image, intensity_image=binary_global)
    
    for region in properties:
        if region['convex_area'] > area_mr_thresh:
            segments = math.ceil(region['convex_area'] * seg_prop)
            compactness = 1

            slices = region['slice']

            '''
            plt.imshow(region['image'], cmap='gray')
            plt.title('segment')
            plt.show()
            '''
            img_x = slices[0].start
            img_y = slices[1].start
            # processor = SLIC(
            #    image=imgRGB[slices], binaryImg=region['image'],  K=segments, M=compactness)
            segments_slic = slic(image=imgRGB[slices], n_segments=segments)
            #segments_slic = processor.execute(iterations=3, labWeight=0.2)
            '''
            plt.imshow(label2rgb(segments_slic))
            plt.title('local labels')
            plt.show()
            '''

            for i in range(segments_slic.shape[0]):
                for j in range(segments_slic.shape[1]):
                    if(region['image'][i][j]):
                        label_image[img_x + i][img_y +
                                               j] = int(segments_slic[i][j]) + n

            # label_image[slices] = [[value + n + 1 if region['image'][x][y] else label_image[x+img_x][y+img_y]
            #                        for y, value in enumerate(row)] for x, row in enumerate(segments_slic)]

            '''
            plt.imshow(label2rgb(label_image))
            plt.title('Label')
            plt.show()'''
            n += segments

    # DEBUG
    '''
    plt.imshow(label2rgb(label=label_image, image=binary_global))
    plt.title('Label')
    plt.show()
    '''

    properties = regionprops(label_image, intensity_image=binary_global)
    for p in properties:
        if p['area'] > area_mr_thresh:
            slices = p['slice']
            h = slices[0].stop-slices[0].start
            w = slices[1].stop-slices[1].start

            rescale_rate = max(h/9, w/9)

            height_rescaled = rescale_rate*41
            width_rescaled = rescale_rate*41

            pad_height = height_rescaled - h
            pad_width = width_rescaled - w

            mask = p['image'].flat
            accountant_indexes = np.where(mask == True)
            counts = np.bincount(label[slices].flat[accountant_indexes[0]])
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
            #connectedComponent = img[p['slice']]

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
            #plt.savefig('imgs/cc/cc'+str(i))
            plt.clf()
            
            plt.imshow(neighbourhood, cmap='gray')
            plt.title('Neighbourhood')
            plt.show()
            #plt.savefig('imgs/nh/nh'+str(i))
            plt.clf()
            '''

            normalized = cv2.resize(neighbourhood, dsize=(
                41, 41), interpolation=cv2.INTER_CUBIC)
            features.append(normalized)

            # DEBUG
            '''
            plt.imshow(normalized, cmap = 'gray')
            plt.title('Normalized')
            #plt.gca().add_patch(Rectangle((15, 15), 9, 9, linewidth=1,
            #                            edgecolor='r', facecolor='none'))
            plt.show()
            #plt.savefig('imgs/normal/normal'+str(i))
            plt.clf()
            '''

    return np.array(features), np.array(labels)


def main():
    f, l = tidy_cc('00000085.jpg', '00000085_label.png')
    print(f.shape)
    print(l.shape)
    print((l == 0).sum())
    print((l == 1).sum())


if __name__ == "__main__":
    main()
