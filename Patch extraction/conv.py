import os
import glob
import numpy as np
from extractor import tidy_cc, tidy_pixels, tidy_superpixels
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
import time
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import random


def conv_cc(optimizer='Adam', loss_func='binary_crossentropy', epochs=20, batch_size=64):
    results = open("experiments/cc/cc.txt", "a+")
    results.write(
        "#####################################################################\n")
    results.write("Parameters:\n")
    results.write("Optimizer: " + optimizer+"\n")
    results.write("Loss: "+loss_func+"\n")
    results.write("Epochs: "+str(epochs)+"\n")
    results.write("Batch_size: "+str(batch_size)+"\n")

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

    results.write("Elapsed time to extract features: " +
                  str(end_t-start_t) + 's\n')
    results.write('Number of features in the training set: ' +
                  str(train_cc.shape[0]) + '\n')
    results.write('Training composition: \n')
    results.write('Number of examples of class 0: ' +
                  str((train_labels == 0).sum()) + '\n')
    results.write('Number of examples of class 1: ' +
                  str((train_labels == 1).sum()) + '\n')
    results.write('Test composition: ' + '\n')
    results.write('Number of examples of class 0: ' +
                  str((test_labels == 0).sum()) + '\n')
    results.write('Number of examples of class 1: ' +
                  str((test_labels == 1).sum()) + '\n')

    train_cc = train_cc.astype('float32') / 255
    test_cc = test_cc.astype('float32') / 255

    # Separates train dataset into classes
    train_index_0 = np.where((train_labels == 0))
    train_index_1 = np.where((train_labels == 1))

    # Chooses randomly 500 samples from each class set
    random_train_index_0 = random.choices(train_index_0[0], k=600)
    random_train_index_1 = random.choices(train_index_1[0], k=600)

    random_train_index = []
    random_train_index += random_train_index_0
    random_train_index += random_train_index_1

    x_train = train_cc[random_train_index]
    y_train = train_labels[random_train_index]

    model.compile(optimizer=optimizer,
                  loss=loss_func,
                  metrics=['accuracy'])
    history = model.fit(x_train[:1000, :], y_train[:1000], epochs=epochs, verbose=0,
                        batch_size=batch_size, validation_data=(x_train[1000:], y_train[1000:]))

    history_dict = history.history
    history_dict.keys()

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(1, len(acc) + 1)

    fig, axarr = plt.subplots(1, 2, figsize=(12, 4))
    # axarr[0].imshow(image_datas[0])
    # "bo" is for "blue dot"
    axarr[0].plot(epochs_range, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    axarr[0].plot(epochs_range, val_loss, 'b', label='Validation loss')

    axarr[0].set_title('Training and validation loss')
    axarr[0].set_xlabel('Epochs')
    axarr[0].set_ylabel('Loss')

    axarr[1].plot(epochs_range, acc, 'bo', label='Training acc')
    axarr[1].plot(epochs_range, val_acc, 'b', label='Validation acc')
    axarr[1].set_title('Training and validation accuracy')
    axarr[1].set_xlabel('Epochs')
    axarr[1].set_ylabel('Loss')

    fig.savefig('experiments/cc/cc_metrics_' + optimizer +
                '_' + loss_func + '_' + str(epochs)+'_'+str(batch_size))

    #test_loss, test_acc = model.evaluate(test_cc, test_labels)

    y_true = test_labels
    y_pred = np.rint(np.squeeze(model.predict(test_cc))).astype('uint8')

    tp = np.sum((y_true == 1)*(y_pred == 1))
    tn = np.sum((y_true == 0)*(y_pred == 0))
    fp = np.sum((y_true == 0)*(y_pred == 1))
    fn = np.sum((y_true == 1)*(y_pred == 0))

    acc = (tp+tn)/float(tp+tn+fp+fn)
    precision = tp/float(tp+fp)
    recall = tp/float(tp+fn)
    results.write('Accuracy: '+str(acc) + '\n')
    results.write('Precision: '+str(precision)+'\n')
    results.write('Recall: '+str(recall)+'\n')

    confusion = [
        [tp, fp],
        [fn, tn]
    ]

    df_cm = pd.DataFrame(confusion, ['$\hat{y} = 1$', '$\hat{y} = 0$'], [
                         '$y = 1$', '$y = 0$'])
    plt.figure(figsize=(10, 7))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap='coolwarm')

    plt.savefig('experiments/cc/cc_confusion_'+optimizer +
                '_'+loss_func+'_'+str(epochs)+'_'+str(batch_size))

    results.write(
        "#####################################################################\n")
    results.close()
    return model


def conv_pixels(optimizer='Adam', loss_func='binary_crossentropy', epochs=20, batch_size=64):
    results = open("experiments/p/p.txt", "a+")
    results.write(
        "#####################################################################\n")
    results.write("Parameters:\n")
    results.write("Optimizer: " + optimizer+"\n")
    results.write("Loss: "+loss_func+"\n")
    results.write("Epochs: "+str(epochs)+"\n")
    results.write("Batch_size: "+str(batch_size)+"\n")

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
        f, l = tidy_pixels(images_train[i], labels_images_train[i], w_size=21)
        train_pixels = np.append(train_pixels, f, axis=0)
        train_labels = np.append(train_labels, l, axis=0)

    for l in range(len(images_test)):
        f, l = tidy_pixels(images_test[l], labels_images_test[l], w_size=21)
        test_pixels = np.append(test_pixels, f, axis=0)
        test_labels = np.append(test_labels, l, axis=0)
    end_t = time.time()


    results.write("Elapsed time to extract features: " +
                  str(end_t-start_t) + 's\n')
    results.write('Number of features in the training set: ' +
                  str(train_pixels.shape[0]) + '\n')
    results.write('Training composition: \n')
    results.write('Number of examples of class 0: ' +
                  str((train_labels == 0).sum()) + '\n')
    results.write('Number of examples of class 1: ' +
                  str((train_labels == 1).sum()) + '\n')
    results.write('Test composition: ' + '\n')
    results.write('Number of examples of class 0: ' +
                  str((test_labels == 0).sum()) + '\n')
    results.write('Number of examples of class 1: ' +
                  str((test_labels == 1).sum()) + '\n')

    train_pixels = train_pixels.astype('float32') / 255
    test_pixels = test_pixels.astype('float32') / 255

    # Separates train dataset into classes
    train_index_0 = np.where((train_labels == 0))
    train_index_1 = np.where((train_labels == 1))

    # Chooses randomly 500 samples from each class set
    random_train_index_0 = random.choices(train_index_0[0], k=600)
    random_train_index_1 = random.choices(train_index_1[0], k=600)

    random_train_index = []
    random_train_index += random_train_index_0
    random_train_index += random_train_index_1

    x_train = train_pixels[random_train_index]
    y_train = train_labels[random_train_index]

    model.compile(optimizer=optimizer,
                  loss=loss_func,
                  metrics=['accuracy'])
    history = model.fit(x_train[:1000, :], y_train[:1000], epochs=epochs, verbose=0,
                        batch_size=batch_size, validation_data=(x_train[1000:], y_train[1000:]))

    history_dict = history.history
    history_dict.keys()

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(1, len(acc) + 1)

    fig, axarr = plt.subplots(1, 2, figsize=(12, 4))
    # axarr[0].imshow(image_datas[0])
    # "bo" is for "blue dot"
    axarr[0].plot(epochs_range, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    axarr[0].plot(epochs_range, val_loss, 'b', label='Validation loss')

    axarr[0].set_title('Training and validation loss')
    axarr[0].set_xlabel('Epochs')
    axarr[0].set_ylabel('Loss')

    axarr[1].plot(epochs_range, acc, 'bo', label='Training acc')
    axarr[1].plot(epochs_range, val_acc, 'b', label='Validation acc')
    axarr[1].set_title('Training and validation accuracy')
    axarr[1].set_xlabel('Epochs')
    axarr[1].set_ylabel('Loss')

    fig.savefig('experiments/p/p_metrics_' + optimizer +
    '_' + loss_func + '_' + str(epochs)+'_'+str(batch_size))

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
    results.write('Accuracy: '+str(acc) + '\n')
    results.write('Precision: '+str(precision)+'\n')
    results.write('Recall: '+str(recall)+'\n')

    confusion = [
        [tp, fp],
        [fn, tn]
    ]

    df_cm = pd.DataFrame(confusion, ['$\hat{y} = 1$', '$\hat{y} = 0$'], [
                         '$y = 1$', '$y = 0$'])
    plt.figure(figsize=(10, 7))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap='coolwarm')

    plt.savefig('experiments/p/p_confusion_'+optimizer +
                '_'+loss_func+'_'+str(epochs)+'_'+str(batch_size))

    results.write(
        "#####################################################################\n")
    results.close()

    return model


def conv_superpixels(optimizer='Adam', loss_func='binary_crossentropy', epochs=20, batch_size=64):
    results = open("experiments/sp/sp.txt", "a+")
    results.write(
        "#####################################################################\n")
    results.write("Parameters:\n")
    results.write("Optimizer: " + optimizer+"\n")
    results.write("Loss: "+loss_func+"\n")
    results.write("Epochs: "+str(epochs)+"\n")
    results.write("Batch_size: "+str(batch_size)+"\n")

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

    results.write("Elapsed time to extract features: " +
                  str(end_t-start_t) + 's\n')
    results.write('Number of features in the training set: ' +
                  str(train_superpixels.shape[0]) + '\n')
    results.write('Training composition: \n')
    results.write('Number of examples of class 0: ' +
                  str((train_labels == 0).sum()) + '\n')
    results.write('Number of examples of class 1: ' +
                  str((train_labels == 1).sum()) + '\n')
    results.write('Test composition: ' + '\n')
    results.write('Number of examples of class 0: ' +
                  str((test_labels == 0).sum()) + '\n')
    results.write('Number of examples of class 1: ' +
                  str((test_labels == 1).sum()) + '\n')

    train_superpixels = train_superpixels.astype('float32') / 255
    test_superpixels = test_superpixels.astype('float32') / 255

    # Separates train dataset into classes
    train_index_0 = np.where((train_labels == 0))
    train_index_1 = np.where((train_labels == 1))

    # Chooses randomly 500 samples from each class set
    random_train_index_0 = random.choices(train_index_0[0], k=600)
    random_train_index_1 = random.choices(train_index_1[0], k=600)

    random_train_index = []
    random_train_index += random_train_index_0
    random_train_index += random_train_index_1

    x_train = train_superpixels[random_train_index]
    y_train = train_labels[random_train_index]

    model.compile(optimizer=optimizer,
                  loss=loss_func,
                  metrics=['accuracy'])
    history = model.fit(x_train[:1000, :], y_train[:1000], epochs=epochs, verbose=0,
                        batch_size=batch_size, validation_data=(x_train[1000:], y_train[1000:]))

    history_dict = history.history
    history_dict.keys()

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(1, len(acc) + 1)

    fig, axarr = plt.subplots(1, 2, figsize=(12, 4))
    # axarr[0].imshow(image_datas[0])
    # "bo" is for "blue dot"
    axarr[0].plot(epochs_range, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    axarr[0].plot(epochs_range, val_loss, 'b', label='Validation loss')

    axarr[0].set_title('Training and validation loss')
    axarr[0].set_xlabel('Epochs')
    axarr[0].set_ylabel('Loss')

    axarr[1].plot(epochs_range, acc, 'bo', label='Training acc')
    axarr[1].plot(epochs_range, val_acc, 'b', label='Validation acc')
    axarr[1].set_title('Training and validation accuracy')
    axarr[1].set_xlabel('Epochs')
    axarr[1].set_ylabel('Loss')

    fig.savefig('experiments/sp/sp_metrics_' + optimizer +
    '_' + loss_func + '_' + str(epochs)+'_'+str(batch_size))

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
    results.write('Accuracy: '+str(acc) + '\n')
    results.write('Precision: '+str(precision)+'\n')
    results.write('Recall: '+str(recall)+'\n')

    confusion = [
        [tp, fp],
        [fn, tn]
    ]

    df_cm = pd.DataFrame(confusion, ['$\hat{y} = 1$', '$\hat{y} = 0$'], [
                         '$y = 1$', '$y = 0$'])
    plt.figure(figsize=(10, 7))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap='coolwarm')

    plt.savefig('experiments/sp/sp_confusion_'+optimizer +
                '_'+loss_func+'_'+str(epochs)+'_'+str(batch_size))

    results.write(
        "#####################################################################\n")
    results.close()

    return model
