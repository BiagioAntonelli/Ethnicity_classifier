import numpy as np
import pandas as pd


import os
from os import listdir
from os.path import isfile, join

import keras

from scipy.ndimage import interpolation
from scipy.misc import imread, imsave, imresize

import cv2
import PIL
from PIL import Image

labels = pd.read_csv(DATA_DIR+"labels.csv")

mypath = DATA_DIR+'training_set'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

X = []
id_img = []
new_size = 128

for i in range(0,len(files)):
    path = mypath+"/"+files[i]
    im = Image.open(path)

    print("Loading img----->"+str(i))
    if len(np.array(im).shape)<3:
        continue
    if np.array(im).shape[0] < new_size or np.array(im).shape[1] < new_size:
        continue

    reduced_size = int(new_size), int(new_size)
    imr = im.resize(reduced_size, Image.ANTIALIAS)
    imr = np.array(imr)#-[103.939, 116.779, 123.68] #vgg centering approach
    X.append(imr)
    id_img.append(files[i]) #store id of the images

X = np.array(X)
id_img = np.array(id_img)

# Function to associate each image to its label
faces = False
y = []
id_list = []
def build_y(id_img,faces = False):
    for i in range(0,len(id_img)):
        sep = '_'
        idx = id_img[i].split(sep, 1)[1]
        if faces:
            sep = "."
            id_temp = idx.split(sep, 1)[0]
            sep = "-"
            idx = id_temp.split(sep, 1)[0]

        id_list.append(idx)
        y_temp = labels.loc[labels['img_id'] == int(idx)].iloc[0,1]
        y.append(y_temp)
    return np.array(y),np.array(id_list)

y,id_list = build_y(id_img,faces=False)

# 1-hot encode
y = pd.Series(y)
hot_y = np.array(pd.get_dummies(y))
pd.get_dummies(y).head()

import pickle
# save to pickle
a = X
filename = 'X_64.pickle'

with open(filename, 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
