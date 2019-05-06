# 
# # Implementação do Artigo do Bukhari
# 
# Aqui farei uma versão de implementação do artigo: "Document Image Segmentation using Discriminative Learning over Connected Components"
# 

import numpy as np 
import scipy.ndimage
from scipy.misc import imresize, toimage, imsave
from skimage import measure
from skimage.measure import label, regionprops
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from sklearn import svm
import os
import pickle
from os.path import basename
from sklearn.externals import joblib
from sklearn.model_selection import KFold

import sys
from CNN_Estimator import CNN_Estimator
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import random
from sklearn.model_selection import train_test_split

images_shape = (40, 40, 1)
num_classes = 2


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0]) , recall_score(labels[:, 0], predictions[:, 0] > 0.5 , average=None),\
  precision_score(labels[:, 0], predictions[:, 0]  > 0.5 , average=None)


def build_model_dir(test_part, features):
    return "model_test_file_%s_features_%s" % (os.path.basename(test_part), os.path.basename(features))

np.set_printoptions(threshold=np.nan)

def split_train_val(x, y):
    return train_test_split(x, y, test_size=0.2, random_state=0)


path = '../../../../anamaia/Data/'#'dados/'

np.set_printoptions(threshold=np.nan)

size = (40,40)
context_proportions = [40/8] #[40/20, 40/16, 40/12, 40/8]
name_proportions = ['8']

# Construindo o ground-truth pelo centro da bounding box
def createGT (properties, mask, GT):
    for region in properties:
        bb = region['bbox']
        cent1 = bb[0] + ((bb[2] - bb[0])//2)
        cent2 = bb[1] + ((bb[3] - bb[1])//2)
        if mask.item(cent1,cent2) == 255:
            GT.append(1)
        else:
            GT.append(0)
    return GT

# Construindo o ground-truth por pixels
def createGTPixel (properties, mask, GT):
    for region in properties:
        img = region['image']
        mask_img = np.zeros(img.shape)
        bb = region['bbox']
        mask_img = mask[bb[0]: bb[2]:, bb[1]: bb[3]:]
        dif = np.count_nonzero(mask_img)/(mask_img.shape[0]*mask_img.shape[1])
        if dif > 0.5:
            GT.append(1)
        else:
            GT.append(0)
    return GT

def features(properties, X, img_train):
    # Chamada das funções de extração de todas as features
    temp = []
    # Features de altura e largura 
    L, H = lengthHeight(properties, img_train)
    print ("L.shape", L.shape)
    print ("H.shape", H.shape)
    # Feature aspect ratio = 
    A = aspectRatio (L, H)
    print ("A.shape", A.shape)
    #P = foregroundPixels (img_train)
    E = elongation (properties)
    print ("E.shape", E.shape)
    S = solidity(properties)
    print ("S.shape", S.shape)
    HU = hu_moments(properties)
    print ("HU.shape", HU.shape)
    XY = x_y_coordinates(properties)
    print ("XY.shape", XY.shape)
    
    temp = H
    temp = np.concatenate((temp, L), axis=1)
    
    temp = np.concatenate((temp, A), axis=1)
    temp = np.concatenate((temp, E), axis=1)
    temp = np.concatenate((temp, S), axis=1)
    temp = np.concatenate((temp, HU), axis=1)
    temp = np.concatenate((temp, XY), axis=1)

    if (len(X) == 0):
        X = temp
    else:
        X = np.concatenate((X, temp), axis=0) 
    return X

def features_ALL(properties, X, img_train):
    # Chamada das funções de extração de todas as features
    temp = []
    aux = []
    # Features de altura e largura 
    L, H = lengthHeight(properties, img_train)
    print ("L.shape", L.shape)
    print ("H.shape", H.shape)
    # Feature aspect ratio = 
    A = aspectRatio (L, H)
    print ("A.shape", A.shape)
    #P = foregroundPixels (img_train)
    E = elongation (properties)
    print ("E.shape", E.shape)
    S = solidity(properties)
    print ("S.shape", S.shape)
    HU = hu_moments(properties)
    print ("HU.shape", HU.shape)
    XY = x_y_coordinates(properties)
    print ("XY.shape", XY.shape)
    
    temp = H
    temp = np.concatenate((temp, L), axis=1)
    
    temp = np.concatenate((temp, A), axis=1)
    temp = np.concatenate((temp, E), axis=1)
    temp = np.concatenate((temp, S), axis=1)
    temp = np.concatenate((temp, HU), axis=1)
    temp = np.concatenate((temp, XY), axis=1)

    aux = featuresIMG_flat(properties,aux, img_train)
    temp = np.concatenate((temp, aux), axis=1)
    aux2 = []
    aux2 = featuresIMG_new_flat(properties, aux2, img_train)
    temp = np.concatenate((temp, aux), axis=1)
    print (temp.shape)

    if (len(X) == 0):
        X = temp
    else:
        X = np.concatenate((X, temp), axis=0) 
    return X

def context (properties, image, w_size):
    #vizinhança da componente
    N = []
    J = []
    
    for region in properties:
        cc_img = region['image']
        new_im = np.zeros(w_size)
        
        nb_size = (2*cc_img.shape[0], 5*cc_img.shape[1])
        nb_img, img_size = contextIMG (nb_size, image, region)
        new_im = rescale(nb_img, w_size)
                
        J.append(np.array(new_im))
        N.append(np.reshape(new_im, (1600)))
        
        #if (region['label'] == 316):
        #    plt.imshow(nb_img)
        #    plt.show()
        #    plt.imshow(new_im)
        #    plt.show()
    return N, J

def contextIMG (nb_size, image, region_cc):
    #Retorna a imagem da vizinhança de tamanho nb_size
    img = region_cc['image']
    min_row = region_cc['bbox'][0]
    max_row = region_cc['bbox'][2]
    min_col = region_cc['bbox'][1]
    max_col = region_cc['bbox'][3]
        
    # Verificação e ajuste caso as novas coordenadas caiam fora da imagem
    new_min_row = min_row - (math.ceil((nb_size[0]-img.shape[0])/2))
    if new_min_row < 0:
        new_min_row = 0
    new_max_row = max_row + (math.floor((nb_size[0]-img.shape[0])/2))
    if new_max_row >= image.shape[0]:
        new_max_row = image.shape[0]
    new_min_col = min_col - (math.ceil((nb_size[1]-img.shape[1])/2))
    if new_min_col < 0:
        new_min_col = 0
    new_max_col = max_col + (math.floor((nb_size[1]-img.shape[1])/2))
    if new_max_col >= image.shape[1]:
        new_max_col = image.shape[1]
        
    actual_size = ((new_max_row - new_min_row), (new_max_col - new_min_col))
    if (actual_size[0] > actual_size[1]):
        sqr_size = (actual_size[0],actual_size[0])
    else:
        sqr_size = (actual_size[1],actual_size[1]) 
            
    nb_img = np.zeros(sqr_size)
        
    ini_l = math.ceil((sqr_size[0] - actual_size[0])/2)
    end_l = math.ceil((sqr_size[0] - actual_size[0])/2) + actual_size[0]
    ini_h = math.ceil((sqr_size[1] - actual_size[1])/2)
    end_h = math.ceil((sqr_size[1] - actual_size[1])/2) + actual_size[1]
        
    nb_img[ini_l:end_l, ini_h: end_h] = image[new_min_row:new_max_row, new_min_col:new_max_col]
        
    return nb_img, sqr_size

def rescale (img, new_size):
    new_im = np.zeros(new_size)
    h,l = img.shape
    nh = new_size[0] #new height
    nl = new_size[1] #new length
    dim = new_size
    if h > 40 and l <= 40:
        sqr_im = np.zeros((h,h))
        sqr_im[0:h, ((h-l)//2):((h-l)//2) + l:] = img
        resizedimg = imresize(sqr_im, new_size, interp='nearest')
    elif h <= 40 and l > 40:
        sqr_im = np.zeros((l,l))
        sqr_im[((l-h)//2):((l-h)//2) + h:,0:l] = img
        resizedimg = imresize(sqr_im, new_size, interp='nearest')
    elif h > 40 and l > 40:
        if (h > l):
            sqr_im = np.zeros((h,h))
            sqr_im[0:h, ((h-l)//2):((h-l)//2) + l:] = img  
        else:
            sqr_im = np.zeros((l,l))
            sqr_im[((l-h)//2):((l-h)//2) + h:,0:l] = img
        resizedimg = imresize(img, new_size, interp='nearest')
    else:
        dim = (h,l)
        resizedimg = img
    new_im[((nh-dim[0])//2):((nh-dim[0])//2) + dim[0]:,((nl-dim[1])//2):((nl-dim[1])//2) + dim[1]:] = resizedimg
    return new_im
        
# Reescalonamento de Componentes
def componentRescale(properties):
        I = []
        X = []
        new_size = (40, 40)
        for region in properties:
            img = region['image']
            new_im = rescale (img, new_size)
            I.append(np.array(new_im))
            X.append(np.reshape(new_im, (1600)))
        X = np.array(X)
        return X,I

def newFeature (properties, image, w_size, i_proportion):
    M = []
    W = []
    
    for region in properties:
        img = region['image']
        
        new_im = np.zeros(w_size)
        
        min_row = region['bbox'][0]
        max_row = region['bbox'][2]
        min_col = region['bbox'][1]
        max_col = region['bbox'][3]
        
        l = max_col - min_col
        h = max_row - min_row
        
        if l > h:
            sqr_size = (l,l)
        else:
            sqr_size = (h,h)
        
        #print ('sqr_size = ', sqr_size)
        
        # Verificação e ajuste caso as novas coordenadas caiam fora da imagem
         # Verificação e ajuste caso as novas coordenadas caiam fora da imagem
        new_min_row = min_row - (math.ceil((sqr_size[0]-img.shape[0])/2))
        if new_min_row < 0:
            new_min_row = 0
        new_max_row = max_row + (math.floor((sqr_size[0]-img.shape[0])/2))
        if new_max_row >= image.shape[0]:
            new_max_row = image.shape[0]
        new_min_col = min_col - (math.ceil((sqr_size[1]-img.shape[1])/2))
        if new_min_col < 0:
            new_min_col = 0
        new_max_col = max_col + (math.floor((sqr_size[1]-img.shape[1])/2))
        if new_max_col >= image.shape[1]:
            new_max_col = image.shape[1]
        
        actual_size = ((new_max_row - new_min_row), (new_max_col - new_min_col))
        
        if (actual_size[0] > actual_size[1]):
            sqr_size = (actual_size[0],actual_size[0])
        else:
            if (actual_size[0] < actual_size[1]):
                sqr_size = (actual_size[1],actual_size[1]) 
        
        nb_size = (math.floor(context_proportions[i_proportion] * sqr_size[0]), math.floor(context_proportions[i_proportion] * sqr_size[1]))

        if (nb_size[0] > nb_size[1]):
            nb_size = (nb_size[0],nb_size[0])
        else:
            if (nb_size[0] < nb_size[1]):
                nb_size = (nb_size[1],nb_size[1])

        nb_img = np.zeros(sqr_size)
        
        ini_l = math.ceil((sqr_size[0] - actual_size[0])/2)
        end_l = math.ceil((sqr_size[0] - actual_size[0])/2) + actual_size[0]
        ini_h = math.ceil((sqr_size[1] - actual_size[1])/2)
        end_h = math.ceil((sqr_size[1] - actual_size[1])/2) + actual_size[1]
        
        nb_img[ini_l:end_l, ini_h: end_h] = image[new_min_row:new_max_row, new_min_col:new_max_col]
        
        db_nb_img, db_nb_size = contextIMG (nb_size, image, region)
        sqr_cc_mask = np.zeros(db_nb_size)
        
        sqr_cc_mask[math.ceil((db_nb_size[0]-img.shape[0])/2):(math.ceil((db_nb_size[0]-img.shape[0])/2))+ img.shape[0],math.ceil((db_nb_size[1]-img.shape[1])/2):(math.ceil((db_nb_size[1]-img.shape[1])/2))+ img.shape[1]] = img * 255
        
         
        db_nb_img = ((db_nb_img - sqr_cc_mask) * -1) + sqr_cc_mask
        new_im = rescale(db_nb_img, w_size)

        
        W.append(np.array(new_im))
        M.append(np.reshape(new_im, (1600)))
 
    return M, W


def featuresIMG(properties, X, img_train):
    # Extração das features em formato de imagem (CNN)
    temp = []

    K, I = componentRescale (properties)
    N, J = context(properties,img_train, size)
    
    temp = I
    temp = np.concatenate((temp, J), axis=1)

    if (len(X) == 0):
        X = temp
    else:
        X = np.concatenate((X, temp), axis=0) 
    return X

def featuresIMG_new(properties, X, img_train, i_proportion):
    # Extração das features em formato de imagem (CNN)
    M, W = newFeature (properties, img_train, size, i_proportion)
    
    if (len(X) == 0):
        X = W
    else:
        X = np.concatenate((X, W), axis=0) 
    return X


# Construindo o conjunto de Treinamento
FTrain = []
file = open('part_1.txt', 'r')
temp = file.read().splitlines()
for i in range(0, len(temp)): 
    FTrain.append(os.path.basename(temp[i]))
imgtrain_in = []
imgtrain_mask = []



train_file_x = 'dados/P2_X_train_8x8.npy'
train_file_y = 'dados/P2_GT.npy'
test_file_y = 'dados/P1_GT.npy'
features = 'both_joined'
if train_file_x == 'dados/P2_X.npy':
    features = 'component_only'
# print(train_file_x)
# x, y = load_data(train_file_x, train_file_y, features)

# x_train, x_val, y_train, y_val = split_train_val(x, y)
# del x, y


learning_rates = [1e-5, 3e-5, 1e-4, 3e-4]
conv_layers = [([5, 5, 1, 32], [5, 5, 32, 64]), ([3, 3, 1, 32], [3, 3, 32, 64])]
conv_strides = [[1, 1, 1, 1], [1, 2, 2, 1]]

# learning_rates = [1e-5, 3e-5]
# conv_layers = [([5, 5, 1, 32], [5, 5, 32, 64])]
# conv_strides = [[1, 3, 3, 1]]

model_dir = build_model_dir(test_file_y, train_file_x)
cnn_estimator = CNN_Estimator(images_shape, num_classes,
                learning_rates=learning_rates, conv_layers=conv_layers, conv_strides=conv_strides,
                model_dir=model_dir, multilabel=False, num_proc=3)

for f in FTrain:
    img = scipy.ndimage.imread(path + "Bin/" + f, mode='L')
    print (f)
    mask = scipy.ndimage.imread(path + "GT/" + os.path.splitext(f)[0][0:-3] + "mask.png", mode='L')
    imgtrain_in.append(img)
    imgtrain_mask.append(mask)

    # Calculando as features do componente e contexto (separadas) para cada imagem. Ambas características são colocadas no mesmo vetor. Teremos um vetor para 
    # cada imagem. O formato do nome do arquivo contendo o vetro é P2_X_train_<nome da imagem>. O ground-truth tem o formato P2_GT_train_<nome da imagem>.
    # for i in range(0, len(imgtrain_in)):
    #     X_train = []
    #     GT_train = []
    #     label_image, ncc = label(imgtrain_in[i],neighbors=8, return_num=True)
    #     properties = regionprops(label_image, imgtrain_in[i])
     
    #     # ### Extrair as features para cada CC.
    #     X_train = featuresIMG(properties, X_train, imgtrain_in[i])
    #     # Construindo o ground-truth
    #     GT_train = createGTPixel(properties, imgtrain_mask[i], GT_train)
    #     GT_train = np.array(GT_train)
    #     X_train = np.array(X_train)
    #     print ("X_train.shape", X_train.shape)
    #     # Salvando os vetores X e y para cada imagem
    #     np.save("P2_X_train_" + os.path.splitext(FTrain[i])[0][0:-4], X_train)
    #     #np.save("P2_newfeature_X", X_train_new)
    #     np.save("P2_GT_train_" + os.path.splitext(FTrain[i])[0][0:-4], GT_train)
    
    # Calculando as features do componente e contexto (juntos na mesma imagem) para cada imagem. Teremos um vetor para 
    # cada imagem. O formato do nome do arquivo contendo o vetro é P2_X_train_new_<nome da imagem>_<proporçao do contexto> 
    # proporção do contexto é a fração equivalente a proporção [40/8, 40/12, 40/16, 40/20] O ground-truth tem o formato 
    # P2_GT_train_new_<nome da imagem>_<proporçao do contexto> .


for j in range (0, len(context_proportions)):
    for i in range(0, len(imgtrain_in)):
        X_train_new = []
        GT_train_new = []
        label_image, ncc = label(imgtrain_in[i],neighbors=8, return_num=True)
        properties = regionprops(label_image, imgtrain_in[i])
     
        # ### Extrair as features para cada CC.
        X_train_new = featuresIMG_new(properties, X_train_new, imgtrain_in[i], j)
        # Construindo o ground-truth
        GT_train_new = createGTPixel(properties, imgtrain_mask[i], GT_train_new)
        GT_train_new = np.array(GT_train_new)
        X_train_new = np.array(X_train_new)

        X_train_new /= 255.
        X_train_new = X_train_new.reshape((-1, 40, 40, 1))

        y_tmp = np.zeros((GT_train_new.shape[0], 2), np.uint8)
        y_tmp[:, 0] = GT_train_new
        y_tmp[:, 1] = 1 - GT_train_new
        # print(os.path.splitext(FTrain[i])[0][0:-4])
        cnn_val = cnn_estimator.predict(X_train_new)
#        np.save(os.path.splitext("test_out/" + FTrain[i])[0][0:-4] + ".npy", cnn_val)
        acc, rec, prec = accuracy(cnn_val, y_tmp)
        print ('%s\t%d\t%f\t%f\t%f\t%f\t%f' % (os.path.splitext(FTrain[i])[0][0:-4], X_train_new.shape[0], acc, 
                rec[0], rec[1], prec[0], prec[1]))

        # print ("X_train_new.shape", X_train_new.shape)
            # # Salvando os vetores X e y para cada imagem
            # np.save("P2_X_train_new_" + os.path.splitext(FTrain[i])[0][0:-4] + '_' + str(name_proportions[j]) , X_train_new)
            # #np.save("P2_newfeature_X", X_train_new)
            # np.save("P2_GT_train_new_" + os.path.splitext(FTrain[i])[0][0:-4] + '_' + str(name_proportions[j]), GT_train_new)
