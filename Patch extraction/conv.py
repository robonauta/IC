import os
import glob
import numpy as np
from extractor import tidy_cc, tidy_pixels, tidy_superpixels
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
import time


def conv_cc():
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

    model.summary()

    cdir = os.getcwd()

    images_train = glob.glob(str(cdir)+"/train/*.jpg")
    labels_images_train = glob.glob(str(cdir)+"/train/*_label.png")

    images_test = glob.glob(str(cdir)+"/test/*.jpg")
    labels_images_test = glob.glob(str(cdir)+"/test/*_label.png")

    images_train.sort()
    images_test.sort()
    labels_images_train.sort()
    labels_images_test.sort()

    train_cc = np.zeros((0, 41, 41, 3), dtype=np.uint8)
    test_cc = np.zeros((0, 41, 41, 3), dtype=np.uint8)
    train_labels = np.zeros((0), dtype=np.uint8)
    test_labels = np.zeros((0), dtype=np.uint8)

    if len(images_train) != len(labels_images_train):
        raise Exception(
            'Number of feature and label images differ at the training set')
    if len(images_test) != len(labels_images_test):
        raise Exception(
            'Number of feature and label images differ at the test set')

    start_t = time.time()
    for i in range(len(images_train)):
        f, l = tidy_cc(images_train[i], labels_images_train[i])
        train_cc = np.append(train_cc, f, axis=0)
        train_labels = np.append(train_labels, l, axis=0)

    for l in range(len(images_test)):
        f, l = tidy_cc(images_test[l], labels_images_test[l])
        test_cc = np.append(test_cc, f, axis=0)
        test_labels = np.append(test_labels, l, axis=0)
    end_t = time.time()

    print("Elapsed time to extract features: " + str(end_t-start_t) + 's')
    print('Number of features in the training set: ' + str(train_cc.shape[0]))

    '''
    (train_images, train_labels) = tidy('00000085.jpg', '00000085_label.png')
    (test_images, test_labels) = tidy('00000088.jpg', '00000088_label.png')
    '''

    train_cc = train_cc.astype('float32') / 255
    test_cc = test_cc.astype('float32') / 255

    # train_labels = to_categorical(train_labels)
    # test_labels = to_categorical(test_labels)

    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_cc, train_labels, epochs=20, batch_size=64)

    test_loss, test_acc = model.evaluate(test_cc, test_labels)

    y_true = test_labels
    y_pred = np.rint(np.squeeze(model.predict(test_cc))).astype('uint8')

    tp = np.sum((y_true == 1)*(y_pred == 1))
    tn = np.sum((y_true == 0)*(y_pred == 0))
    fp = np.sum((y_true == 0)*(y_pred == 1))
    fn = np.sum((y_true == 1)*(y_pred == 0))

    acc = (tp+tn)/float(tp+tn+fp+fn)
    precision = tp/float(tp+fp)
    recall = tp/float(tp+fn)
    print('Accuracy: '+str(acc))
    print('Precision: '+str(precision))
    print('Recall: '+str(recall))


def conv_pixels():
    model = models.Sequential()
    model.add(layers.Conv2D(
        32, (3, 3), activation='relu', input_shape=(21, 21, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()

    cdir = os.getcwd()

    images_train = glob.glob(str(cdir)+"/train/*.jpg")
    labels_images_train = glob.glob(str(cdir)+"/train/*_label.png")

    images_test = glob.glob(str(cdir)+"/test/*.jpg")
    labels_images_test = glob.glob(str(cdir)+"/test/*_label.png")

    images_train.sort()
    images_test.sort()
    labels_images_train.sort()
    labels_images_test.sort()

    train_pixels = np.zeros((0, 21, 21, 3), dtype=np.uint8)
    test_pixels = np.zeros((0, 21, 21, 3), dtype=np.uint8)
    train_labels = np.zeros((0), dtype=np.uint8)
    test_labels = np.zeros((0), dtype=np.uint8)

    if len(images_train) != len(labels_images_train):
        raise Exception(
            'Number of feature and label images differ at the training set')
    if len(images_test) != len(labels_images_test):
        raise Exception(
            'Number of feature and label images differ at the test set')

    start_t = time.time()
    for i in range(len(images_train)):
        print(images_train[i])
        f, l = tidy_pixels(images_train[i], labels_images_train[i], w_size=21)
        train_pixels = np.append(train_pixels, f, axis=0)
        train_labels = np.append(train_labels, l, axis=0)

    for l in range(len(images_test)):
        f, l = tidy_pixels(images_test[l], labels_images_test[l], w_size=21)
        test_pixels = np.append(test_pixels, f, axis=0)
        test_labels = np.append(test_labels, l, axis=0)
    end_t = time.time()

    print("Elapsed time to extract features: " + str(end_t-start_t) + 's')
    print('Number of features in the training set: ' +
          str(train_pixels.shape[0]))

    print("Distribution of classes in the training set: ")
    print('0: ' + str((train_labels == 0).sum()))
    print('1: ' + str((train_labels == 1).sum()))
    print("Distribution of classes in the test set: ")
    print('0: ' + str((test_labels == 0).sum()))
    print('1: ' + str((test_labels == 1).sum()))

    '''
    (train_images, train_labels) = tidy('00000085.jpg', '00000085_label.png')
    (test_images, test_labels) = tidy('00000088.jpg', '00000088_label.png')
    '''

    train_pixels = train_pixels.astype('float32') / 255
    test_pixels = test_pixels.astype('float32') / 255

    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_pixels, train_labels, epochs=20, batch_size=64)

    #test_loss, test_acc = model.evaluate(test_pixels, test_labels)

    y_true = test_labels
    y_pred = np.rint(np.squeeze(model.predict(test_pixels))).astype('uint8')

    tp = np.sum((y_true == 1)*(y_pred == 1))
    tn = np.sum((y_true == 0)*(y_pred == 0))
    fp = np.sum((y_true == 0)*(y_pred == 1))
    fn = np.sum((y_true == 1)*(y_pred == 0))

    acc = (tp+tn)/float(tp+tn+fp+fn)
    precision = tp/float(tp+fp)
    recall = tp/float(tp+fn)
    print('Accuracy: '+str(acc))
    print('Precision: '+str(precision))
    print('Recall: '+str(recall))


def conv_superpixels():
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

    model.summary()

    cdir = os.getcwd()

    images_train = glob.glob(str(cdir)+"/train/*.jpg")
    labels_images_train = glob.glob(str(cdir)+"/train/*_label.png")

    images_test = glob.glob(str(cdir)+"/test/*.jpg")
    labels_images_test = glob.glob(str(cdir)+"/test/*_label.png")

    images_train.sort()
    images_test.sort()
    labels_images_train.sort()
    labels_images_test.sort()

    train_superpixels = np.zeros((0, 41, 41, 3), dtype=np.uint8)
    test_superpixels = np.zeros((0, 41, 41, 3), dtype=np.uint8)
    train_labels = np.zeros((0), dtype=np.uint8)
    test_labels = np.zeros((0), dtype=np.uint8)

    if len(images_train) != len(labels_images_train):
        raise Exception(
            'Number of feature and label images differ at the training set')
    if len(images_test) != len(labels_images_test):
        raise Exception(
            'Number of feature and label images differ at the test set')

    start_t = time.time()
    for i in range(len(images_train)):
        f, l = tidy_superpixels(images_train[i], labels_images_train[i])
        train_superpixels = np.append(train_superpixels, f, axis=0)
        train_labels = np.append(train_labels, l, axis=0)

    for l in range(len(images_test)):
        f, l = tidy_superpixels(images_test[l], labels_images_test[l])
        test_superpixels = np.append(test_superpixels, f, axis=0)
        test_labels = np.append(test_labels, l, axis=0)
    end_t = time.time()

    print("Elapsed time to extract features: " + str(end_t-start_t) + 's')
    print('Number of features in the training set: ' +
          str(train_superpixels.shape))

    print("Distribution of classes in the training set: ")
    print('0: ' + str((train_labels == 0).sum()))
    print('1: ' + str((train_labels == 1).sum()))
    print("Distribution of classes in the test set: ")
    print('0: ' + str((test_labels == 0).sum()))
    print('1: ' + str((test_labels == 1).sum()))

    '''
    (train_images, train_labels) = tidy('00000085.jpg', '00000085_label.png')
    (test_images, test_labels) = tidy('00000088.jpg', '00000088_label.png')
    '''

    train_superpixels = train_superpixels.astype('float32') / 255
    test_superpixels = test_superpixels.astype('float32') / 255

    # train_labels = to_categorical(train_labels)
    # test_labels = to_categorical(test_labels)

    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_superpixels, train_labels, epochs=20, batch_size=64)

    #test_loss, test_acc = model.evaluate(test_superpixels, test_labels)

    y_true = test_labels
    y_pred = np.rint(np.squeeze(model.predict(
        test_superpixels))).astype('uint8')

    tp = np.sum((y_true == 1)*(y_pred == 1))
    tn = np.sum((y_true == 0)*(y_pred == 0))
    fp = np.sum((y_true == 0)*(y_pred == 1))
    fn = np.sum((y_true == 1)*(y_pred == 0))

    acc = (tp+tn)/float(tp+tn+fp+fn)
    precision = tp/float(tp+fp)
    recall = tp/float(tp+fn)
    print('Accuracy: '+str(acc))
    print('Precision: '+str(precision))
    print('Recall: '+str(recall))
