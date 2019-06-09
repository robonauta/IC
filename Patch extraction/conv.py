import os
import glob
import numpy as np
from extractor import tidy_cc
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical

def conv_cc():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(41, 41, 3)))
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

    train_cc = np.zeros((0,41,41,3), dtype=np.uint8)
    test_cc = np.zeros((0,41,41,3), dtype=np.uint8)
    train_labels = np.zeros((0), dtype=np.uint8)
    test_labels = np.zeros((0), dtype=np.uint8)

    if len(images_train) != len(labels_images_train):
        raise Exception(
            'Number of feature and label images differ at the training set')
    if len(images_test) != len(labels_images_test):
        raise Exception(
            'Number of feature and label images differ at the test set')

    for i in range(len(images_train)):
        f, l = tidy_cc(images_train[i], labels_images_train[i])
        train_cc = np.append(train_cc, f, axis=0)
        train_labels = np.append(train_labels, l, axis=0)

    for l in range(len(images_test)):
        f, l = tidy_cc(images_test[l], labels_images_test[l])
        test_cc = np.append(test_cc, f, axis=0)
        test_labels = np.append(test_labels, l, axis=0)

    '''
    (train_images, train_labels) = tidy('00000085.jpg', '00000085_label.png')
    (test_images, test_labels) = tidy('00000088.jpg', '00000088_label.png')
    '''

    train_cc = train_cc.astype('float32') / 255
    test_cc = test_cc.astype('float32') / 255

    #train_labels = to_categorical(train_labels)
    #test_labels = to_categorical(test_labels)

    model.compile(optimizer='Adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    model.fit(train_cc, train_labels, epochs=20, batch_size=64)


    test_loss, test_acc = model.evaluate(test_cc, test_labels)

    print(test_acc)

def conv_pixels():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(41, 41, 3)))
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

    train_pixels = np.zeros((0,41,41,3), dtype=np.uint8)
    test_pixels = np.zeros((0,41,41,3), dtype=np.uint8)
    train_labels = np.zeros((0), dtype=np.uint8)
    test_labels = np.zeros((0), dtype=np.uint8)

    if len(images_train) != len(labels_images_train):
        raise Exception(
            'Number of feature and label images differ at the training set')
    if len(images_test) != len(labels_images_test):
        raise Exception(
            'Number of feature and label images differ at the test set')

    for i in range(len(images_train)):
        f, l = tidy_cc(images_train[i], labels_images_train[i])
        train_pixels = np.append(train_pixels, f, axis=0)
        train_labels = np.append(train_labels, l, axis=0)

    for l in range(len(images_test)):
        f, l = tidy_cc(images_test[l], labels_images_test[l])
        test_pixels = np.append(test_pixels, f, axis=0)
        test_labels = np.append(test_labels, l, axis=0)

    '''
    (train_images, train_labels) = tidy('00000085.jpg', '00000085_label.png')
    (test_images, test_labels) = tidy('00000088.jpg', '00000088_label.png')
    '''

    train_cc = train_cc.astype('float32') / 255
    test_cc = test_cc.astype('float32') / 255

    #train_labels = to_categorical(train_labels)
    #test_labels = to_categorical(test_labels)

    model.compile(optimizer='Adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    model.fit(train_pixels, train_labels, epochs=20, batch_size=64)


    test_loss, test_acc = model.evaluate(test_cc, test_labels)

    print(test_acc)
