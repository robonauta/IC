import os
import glob
import numpy as np
import math

from extract import tidy_cc, tidy_pixels, tidy_superpixels
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
from skimage import measure
from skimage.color import rgb2gray, label2rgb
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage.util import img_as_float, img_as_uint, img_as_float32, invert
import time
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import random
import cv2

from skimage.segmentation import slic

from extract import consider_cc_thresh, split_cc_thresh


def train(granularity, optimizer='Adam', loss_func='binary_crossentropy', epochs=20, batch_size=64, save=False, set_distribution=0):
    if granularity != 'cc' and granularity != 'sp' and granularity != 'p':
        raise Exception(
            'Granularity must be cc (connected components), sp (superpixels) or p (pixels).')
    if set_distribution < 0 or set_distribution > 2:
        raise Exception(
            'Set distribution must be 0 (equilibrated), 1 (original proportions) or 2 (twice more text than non-text).')

    # Writes parameters used
    results = open("experiments/"+granularity+"/val_"+granularity+".txt", "a+")
    results.write(
        "#####################################################################\n")
    results.write("Parameters:\n")
    results.write("Optimizer: " + optimizer+"\n")
    results.write("Loss: "+loss_func+"\n")
    results.write("Epochs: "+str(epochs)+"\n")
    results.write("Batch_size: "+str(batch_size)+"\n")

    print('Training and validating with parameters:')
    print("Optimizer: " + optimizer)
    print("Loss: "+loss_func)
    print("Epochs: "+str(epochs))
    print("Batch_size: "+str(batch_size))

    # Builds the model
    model = models.Sequential()
    model.add(layers.Conv2D(
        32, (3, 3), activation='relu', input_shape=(41, 41, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # model.summary()

    # Gets currect directory path
    cdir = os.getcwd()

    # Gets all files .jpg from the folder 'train'
    images_train = glob.glob(str(cdir)+"/train/*.jpg")

    # Gets all files _label.png from the folder 'train'
    labels_images_train = glob.glob(str(cdir)+"/train/*_label.png")

    # Puts labels and features in the same order in the array
    images_train.sort()
    labels_images_train.sort()

    # Builds arrays that will host all the patches and labels
    train_examples = np.zeros((0, 41, 41, 3), dtype=np.uint8)
    train_labels = np.zeros((0), dtype=np.uint8)

    # To check if every image has its label correspondence
    if len(images_train) != len(labels_images_train):
        raise Exception(
            'Number of feature and label images differ at the training set')

    start_t = time.time()
    # Gets all the patches and labels for each of the images
    for i in range(len(images_train)):
        f = None
        l = None
        if granularity == 'cc':
            f, l = tidy_cc(images_train[i], labels_images_train[i])
        elif granularity == 'sp':
            f, l = tidy_superpixels(images_train[i], labels_images_train[i])
        else:
            f, l = tidy_pixels(images_train[i], labels_images_train[i])
        train_examples = np.append(train_examples, f, axis=0)
        train_labels = np.append(train_labels, l, axis=0)
    end_t = time.time()

    # Writes some stats
    results.write("Elapsed time to extract features: " +
                  str(end_t-start_t) + 's\n')
    results.write('Number of features in the training set: ' +
                  str(train_examples.shape[0]) + '\n')
    results.write('Extracted patches composition: \n')
    class_0 = (train_labels == 0).sum()
    class_1 = (train_labels == 1).sum()
    results.write('Number of examples of class 0: ' +
                  str(class_0) + '\n')
    results.write('Number of examples of class 1: ' +
                  str(class_1) + '\n')

    train_examples = train_examples.astype('float32') / 255

    # Gets 20% of each class in their original proportions to
    # build a validation set.
    val_k_0 = int(class_0*0.2)
    val_k_1 = int(class_1*0.2)

    if set_distribution == 0:
        # The class proportion in the training set must be equal
        train_k_0 = min(class_0 - val_k_0, class_1-val_k_1)
        train_k_1 = min(class_0 - val_k_0, class_1-val_k_1)
    elif set_distribution == 1:
        # The class proportion in the training set must be the original
        # one
        train_k_0 = class_0 - val_k_0
        train_k_1 = class_1 - val_k_1
    else:
        train_k_0 = class_0 - val_k_0
        train_k_1 = class_1 - val_k_1
        while(train_k_1/2.0 > train_k_0):
            train_k_1 /= 2.0
        train_k_0 = int(train_k_1/2)

    train_k_0 = int(train_k_0)
    train_k_1 = int(train_k_1)

    # Separates train dataset into classes
    train_index_0 = np.where((train_labels == 0))
    train_index_1 = np.where((train_labels == 1))

    # Chooses randomly the validation set according to the real proportion
    # of the classes 0 and 1 in the training set.
    random_val_index_0 = random.choices(train_index_0[0], k=val_k_0)
    random_val_index_1 = random.choices(train_index_1[0], k=val_k_1)

    random_val_index = []
    random_val_index += random_val_index_0
    random_val_index += random_val_index_1
    random_val_index.sort()

    x_val = train_examples[random_val_index]
    y_val = train_labels[random_val_index]

    # Remove the already taken indexes for the validation class, to avoid
    # that a element appears simultaneously in the train and in the validation
    train_index_0 = np.setdiff1d(train_index_0[0], random_val_index_0)
    train_index_1 = np.setdiff1d(train_index_1[0], random_val_index_1)

    # Chooses randomly samples from each class set
    random_train_index_0 = random.choices(train_index_0, k=train_k_0)
    random_train_index_1 = random.choices(train_index_1, k=train_k_1)

    # Merges random indexes
    random_train_index = []
    random_train_index += random_train_index_0
    random_train_index += random_train_index_1
    random_train_index.sort()

    # Gets only the desired random examples and labels
    x_train = train_examples[random_train_index]
    y_train = train_labels[random_train_index]

    results.write('Training set composition: \n')
    results.write('Class 0: '+str((y_train == 0).sum())+' \n')
    results.write('Class 1: '+str((y_train == 1).sum())+'\n')
    results.write('Validation set composition: \n')
    results.write('Class 0: '+str((y_val == 0).sum())+'\n')
    results.write('Class 1: '+str((y_val == 1).sum())+'\n')

    model.compile(optimizer=optimizer,
                  loss=loss_func,
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=epochs, verbose=0,
                        batch_size=batch_size, validation_data=(x_val, y_val))

    # Plots results
    history_dict = history.history
    history_dict.keys()

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(1, len(acc) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epochs_range, loss, 'bo', label='Training loss')
    ax[0].plot(epochs_range, val_loss, 'b', label='Validation loss')

    ax[0].set_title('Training and validation loss for ' + granularity)
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')

    ax[1].plot(epochs_range, acc, 'bo', label='Training acc')
    ax[1].plot(epochs_range, val_acc, 'b', label='Validation acc')
    ax[1].set_title('Training and validation accuracy for ' + granularity)
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')

    fig.savefig('experiments/'+granularity+'/metrics_'+granularity+'_' + optimizer +
                '_' + loss_func + '_' + str(epochs)+'_'+str(batch_size)+'_'+str(set_distribution))

    results.write(
        "#####################################################################\n")
    results.close()

    if save == True:
        model.save('experiments/'+granularity+'/model_'+granularity+'_'+optimizer +
                   '_'+loss_func+'_'+str(epochs)+'_'+str(batch_size)+'_'+str(set_distribution)+'.h5')

    return model

def test():
    # Gets the current path
    cdir = os.getcwd()

    images_test = glob.glob(str(cdir)+"/test/*.jpg")
    labels_images_test = glob.glob(str(cdir)+"/test/*_label.png")

    images_test.sort()
    labels_images_test.sort()

    granularities = ['cc','sp']

    model_filenames = {
        "cc": "experiments/cc/model_cc_Adam_binary_crossentropy_20_256_2.h5",
        "sp": "experiments/sp/model_sp_Adam_binary_crossentropy_20_256_0.h5",
        "p": "model_p_Adam_binary_crossentropy_20_1028_0.h5"
    }

    for g in granularities:
        results = open("experiments/"+g+"_test.txt", "a+")
        results.write(
            "#####################################################################\n")
        results.write('Model: ' + model_filenames[g] + '\n')

        # Those metrics ending with _p are in terms of images' pixels.
        total_tp_p = 0
        total_tn_p = 0
        total_fp_p = 0
        total_fn_p = 0

        # Those ones ending with _r are in terms of images' regions.
        total_tp_r = 0
        total_tn_r = 0
        total_fp_r = 0
        total_fn_r = 0

        for i in range(len(images_test)):
            if g == 'cc':
                result = test_on_img_cc(
                    model_filenames[g], images_test[i], labels_images_test[i], save_predict=True)
                tp_p, tn_p, fp_p, fn_p = result[0]
                tp_r, tn_r, fp_r, fn_r = result[1]
            elif g == 'sp':
                result = test_on_img_sp(
                    model_filenames[g], images_test[i], labels_images_test[i], save_predict=True)
                tp_p, tn_p, fp_p, fn_p = result[0]
                tp_r, tn_r, fp_r, fn_r = result[1]
            elif g == 'p':
                result = test_on_img_p(
                    model_filenames[g], images_test[i], labels_images_test[i], save_predict=True)
                tp_p, tn_p, fp_p, fn_p = result[0]
                tp_r, tn_r, fp_r, fn_r = result[1]

            # Calculates accuracy, precision and recall in terms of pixels
            acc_p = (tp_p+tn_p)/float(tp_p+tn_p+fp_p+fn_p)
            precision_p = tp_p/float(tp_p+fp_p)
            recall_p = tp_p/float(tp_p+fn_p)

            # Calculates accuracy, precision and recall in terms of regions
            acc_r = (tp_r+tn_r)/float(tp_r+tn_r+fp_r+fn_r)
            precision_r = tp_r/float(tp_r+fp_r)
            recall_r = tp_r/float(tp_r+fn_r)

            # Writes metrics
            results.write(images_test[i][-12:] + '\n')
            results.write("         IN TERMS OF PIXELS: \n")
            results.write("         tp: " + str(tp_p) + '\n')
            results.write("         tn: " + str(tn_p) + '\n')
            results.write("         fp: " + str(fp_p) + '\n')
            results.write("         fn: " + str(fn_p) + '\n')
            results.write('         accuracy: '+str(acc_p) + '\n')
            results.write('         precision: '+str(precision_p)+'\n')
            results.write('         recall: '+str(recall_p)+'\n')
            results.write("         IN TERMS OF REGIONS: \n")
            results.write("         tp: " + str(tp_r) + '\n')
            results.write("         tn: " + str(tn_r) + '\n')
            results.write("         fp: " + str(fp_r) + '\n')
            results.write("         fn: " + str(fn_r) + '\n')
            results.write('         accuracy: '+str(acc_r) + '\n')
            results.write('         precision: '+str(precision_r)+'\n')
            results.write('         recall: '+str(recall_r)+'\n')
            total_tp_p += tp_p
            total_tn_p += tn_p
            total_fp_p += fp_p
            total_fn_p += fn_p
            total_tp_r += tp_r
            total_tn_r += tn_r
            total_fp_r += fp_r
            total_fn_r += fn_r

        results.write(
            "#####################################################################\n")

        # Calculates total accuracy, precision and recall in terms of pixels
        total_acc_p = (total_tp_p+total_tn_p) / \
            float(total_tp_p+total_tn_p+total_fp_p+total_fn_p)
        total_precision_p = total_tp_p/float(total_tp_p+total_fp_p)
        total_recall_p = total_tp_p/float(total_tp_p+total_fn_p)

        # Calculates total accuracy, precision and recall in terms of pixels
        total_acc_r = (total_tp_r+total_tn_r) / \
            float(total_tp_r+total_tn_r+total_fp_r+total_fn_r)
        total_precision_r = total_tp_r/float(total_tp_r+total_fp_r)
        total_recall_r = total_tp_r/float(total_tp_r+total_fn_r)

        results.write('Totals IN TERMS OF PIXELS: \n')
        results.write("tp: " + str(total_tp_p) + '\n')
        results.write("tn: " + str(total_tn_p) + '\n')
        results.write("fp: " + str(total_fp_p) + '\n')
        results.write("fn: " + str(total_fn_p) + '\n')

        results.write('accuracy: '+str(total_acc_p) + '\n')
        results.write('precision: '+str(total_precision_p)+'\n')
        results.write('recall: '+str(total_recall_p)+'\n')

        results.write('Totals IN TERMS OF REGIONS: \n')
        results.write("tp: " + str(total_tp_r) + '\n')
        results.write("tn: " + str(total_tn_r) + '\n')
        results.write("fp: " + str(total_fp_r) + '\n')
        results.write("fn: " + str(total_fn_r) + '\n')

        results.write('accuracy: '+str(total_acc_r) + '\n')
        results.write('precision: '+str(total_precision_r)+'\n')
        results.write('recall: '+str(total_recall_r)+'\n')

        results.write(
            "#####################################################################\n")
        results.close()

        plt.clf()
        # Draws confusion matrix
        confusion_p = [
            [total_tp_p, total_fp_p],
            [total_fn_p, total_tn_p]
        ]

        df_cm = pd.DataFrame(confusion_p, ['$\hat{y} = 1$', '$\hat{y} = 0$'], [
            '$y = 1$', '$y = 0$'])
        plt.figure(figsize=(10, 7))
        sn.set(font_scale=1.2)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 14}, cmap='coolwarm')

        plt.savefig('experiments/'+g+'_confusion_p.png')

        plt.clf()
        # Draws confusion matrix
        confusion_r = [
            [total_tp_r, total_fp_r],
            [total_fn_r, total_tn_r]
        ]

        df_cm = pd.DataFrame(confusion_r, ['$\hat{y} = 1$', '$\hat{y} = 0$'], [
            '$y = 1$', '$y = 0$'])
        plt.figure(figsize=(10, 7))
        sn.set(font_scale=1.2)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 14}, cmap='coolwarm')

        plt.savefig('experiments/'+g+'_confusion_r.png')

        plt.clf()

def test_on_img_cc(model_filename, imgRGB_filename, label_img_filename, save_predict=False):
    model = models.load_model(model_filename)

    imgRGB = cv2.imread(imgRGB_filename)
    img = rgb2gray(imgRGB)
    img = img_as_uint(img)

    true_label = cv2.imread(label_img_filename, 0)

    global_thresh = threshold_otsu(img)
    binary_global = img < global_thresh

    label_image = measure.label(binary_global, neighbors=8, background=0)

    properties = regionprops(label_image, intensity_image=binary_global)

    result = np.zeros(imgRGB.shape, dtype=np.uint8)

    tp_r = 0
    tn_r = 0
    fp_r = 0
    fn_r = 0

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

        l = np.rint(np.squeeze(l)).astype('uint8')

        # Counts the most frequent label over the region
        mask = p['image'].flat
        accountant_indexes = np.where(mask == True)
        counts = np.bincount(
            true_label[slices].flat[accountant_indexes[0]])
        label_value = np.argmax(counts)

        # and check if the prediction for that region is correct
        if label_value == 1 and l == 1:
            tp_r += 1
        elif label_value == 0 and l == 0:
            tn_r += 1
        elif label_value == 0 and l == 1:
            fp_r += 1
        elif label_value == 1 and l == 0:
            fn_r += 1

        cc_index = np.where(p['image'])
        result[p['slice']][cc_index] = l
    
    if(save_predict == True):
        cv2.imwrite('experiments/cc/predicted_' +
                    imgRGB_filename[-12:-4]+'.png', result*255)

    tp_p = np.sum((true_label == 1)*(result[:, :, 0] == 1))
    tn_p = np.sum((true_label == 0)*(result[:, :, 0] == 0))
    fp_p = np.sum((true_label == 0)*(result[:, :, 0] == 1))
    fn_p = np.sum((true_label == 1)*(result[:, :, 0] == 0))

    # Those metrics ending with _p are in terms of images' pixels.
    # Those ones ending with _r are in terms of images' regions.
    return [[tp_p, tn_p, fp_p, fn_p], [tp_r, tn_r, fp_r, fn_r]]

def test_on_img_p(model_filename, imgRGB_filename, label_img_filename, save_predict=False):
    model = models.load_model(model_filename)

    imgRGB = cv2.imread(imgRGB_filename)
    img = rgb2gray(imgRGB)
    img = img_as_uint(img)

    true_label = cv2.imread(label_img_filename, 0)

    global_thresh = threshold_otsu(img)
    binary_global = img < global_thresh

    label_image = measure.label(binary_global, neighbors=8, background=0)

    properties = regionprops(label_image, intensity_image=binary_global)

    result = np.zeros(imgRGB.shape, dtype=np.uint8)

    w_size = 41
    for x in range(imgRGB.shape[0]):
        for y in range(imgRGB.shape[1]):
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

            l = model.predict(neighbourhood.reshape(
                (1, neighbourhood.shape[0], neighbourhood.shape[1], neighbourhood.shape[2])))
            result[x][y] = np.rint(np.squeeze(l)).astype('uint8')

    if(save_predict == True):
        cv2.imwrite('experiments/p/predicted_' +
                    imgRGB_filename[-12:-4]+'.png', result*255)

    tp = np.sum((true_label == 1)*(result[:, :, 0] == 1))
    tn = np.sum((true_label == 0)*(result[:, :, 0] == 0))
    fp = np.sum((true_label == 0)*(result[:, :, 0] == 1))
    fn = np.sum((true_label == 1)*(result[:, :, 0] == 0))

    return [tp, tn, fp, fn]

def test_on_img_sp(model_filename, imgRGB_filename, label_img_filename, save_predict=False):
    model = models.load_model(model_filename)

    w_size = 41

    # Opens image
    imgRGB = cv2.imread(imgRGB_filename)
    img = rgb2gray(imgRGB)
    img = img_as_uint(img)

    true_label = cv2.imread(label_img_filename, 0)

    # Binarizes image
    global_thresh = threshold_otsu(img)
    binary_global = img < global_thresh

    # Calculates connected components
    label_image, n = measure.label(
        binary_global, neighbors=8, background=0, return_num=True)

    seg_prop = 0.001

    result = np.zeros(imgRGB.shape, dtype=np.uint8)

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

    tp_r = 0
    tn_r = 0
    fp_r = 0
    fn_r = 0

    # Iterates through CCs and superpixels
    for p in properties:
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
        l = model.predict(normalized.reshape(
            (1, normalized.shape[0], normalized.shape[1], normalized.shape[2])))

        l = np.rint(np.squeeze(l)).astype('uint8')

        # Counts the most frequent label over the region
        mask = p['image'].flat
        accountant_indexes = np.where(mask == True)
        counts = np.bincount(
            true_label[slices].flat[accountant_indexes[0]])
        label_value = np.argmax(counts)

        # and check if the prediction for that region is correct
        if label_value == 1 and l == 1:
            tp_r += 1
        elif label_value == 0 and l == 0:
            tn_r += 1
        elif label_value == 0 and l == 1:
            fp_r += 1
        elif label_value == 1 and l == 0:
            fn_r += 1

        cc_index = np.where(p['image'])
        result[p['slice']][cc_index] = l
    
    if(save_predict == True):
        cv2.imwrite('experiments/sp/predicted_' +
                    imgRGB_filename[-12:-4]+'.png', result*255)

    tp_p = np.sum((true_label == 1)*(result[:, :, 0] == 1))
    tn_p = np.sum((true_label == 0)*(result[:, :, 0] == 0))
    fp_p = np.sum((true_label == 0)*(result[:, :, 0] == 1))
    fn_p = np.sum((true_label == 1)*(result[:, :, 0] == 0))

    # Those metrics ending with _p are in terms of images' pixels.
    # Those ones ending with _r are in terms of images' regions.
    return [[tp_p, tn_p, fp_p, fn_p], [tp_r, tn_r, fp_r, fn_r]]

def main():
    '''
    Grid search code 
    op = ['Adadelta', 'Adam']
    loss = ['binary_crossentropy']
    gran = ['cc', 'p', 'sp']
    epochs = [20]
    b_size = [64, 128, 256, 512, 1028]

        0 - equilibrated
        1 - original proportions
        2 - twice text than non-text

    set_distribution = [0, 1, 2]

    for o in op:
        for l in loss:
            for g in gran:
                for e in epochs:
                    for b in b_size:
                        for s in set_distribution:
                            train(granularity=g, optimizer=o, loss_func=l,
                                  epochs=e, batch_size=b, save=True, set_distribution=s)
    '''

    #train(granularity='cc', optimizer='Adam', loss_func='binary_crossentropy',
    #      epochs=20, batch_size=256, save=True, set_distribution=2)
    test()


if __name__ == '__main__':
    main()
