#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import SimpleITK as sitk


# # DATA
# ## GET DATA

from glob import glob
from typing import  List


BUY_file_paths =  sorted(glob("./PREPROCESSED/BUY/*.nii.gz"))  #68 files
EAT_file_paths =  sorted(glob("./PREPROCESSED/EAT/*.nii.gz"))  #files
GAMBLE_file_paths =  sorted(glob("./PREPROCESSED/GAMBLE/*.nii.gz"))  #7 files
SEX_file_paths =  sorted(glob("./PREPROCESSED/SEX/*.nii.gz"))  #42 files

PD_file_paths =  sorted(glob("./PREPROCESSED/PD/*.nii.gz"))  #100 files


print(len(BUY_file_paths), BUY_file_paths[:3])
print(len(EAT_file_paths), EAT_file_paths[:3])
print(len(GAMBLE_file_paths), GAMBLE_file_paths[:3])
print(len(SEX_file_paths), SEX_file_paths[:3])

print(len(PD_file_paths), PD_file_paths[:3])

X_dataset = []
y_dataset = []

X_dataset.extend(PD_file_paths), y_dataset.extend([1] * len(PD_file_paths))
X_dataset.extend(BUY_file_paths), y_dataset.extend([2] * len(BUY_file_paths))
X_dataset.extend(EAT_file_paths), y_dataset.extend([3] * len(EAT_file_paths))
X_dataset.extend(GAMBLE_file_paths), y_dataset.extend([4] * len(GAMBLE_file_paths))
X_dataset.extend(SEX_file_paths), y_dataset.extend([5] * len(SEX_file_paths))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, test_size = 0.2, stratify=y_dataset)

print("Dataset:", len(X_dataset), len(y_dataset))
print("Train:", len(X_train), len(y_train))
print("Test:", len(X_test), len(y_test))


# ## PREPROSSESING

def get_category(category, is_binary = True):
  result = np.zeros(2 if is_binary else 5)
  if is_binary:
    return 1 if category == 1 else 0
  elif category == 1:
    result[0] = 1
  else:
    result[category-1] = 1
  return result

def get_categories(y_data):
  categories = None
  is_first = True
  for category in y_data:
    if is_first:
      categories = np.array([get_category(category)])
      is_first = False
    else:
      categories = np.concatenate((categories,[get_category(category)]))
  return categories

print(get_category(5, True), get_category(5, False))
print(get_category(1, True), get_category(1, False))


# ## 3D All Brain

import cv2

dimension = 180
def get_slices_3d_all(path, category):
    img = sitk.ReadImage(path, sitk.sitkFloat64)
    arr = sitk.GetArrayFromImage(img)

    #normalize the matrix, numbers between 0.0 - 1.0
    arr = arr / arr.max()
    
    slice = arr[10:160, : , :]
    
    arr = np.zeros([150, dimension,dimension])
    
    for i in range(arr.shape[0]):
        arr[i, : , :] = cv2.resize(slice[i, : , :], (dimension, dimension), interpolation=cv2.INTER_CUBIC)
    
    slices = np.array([arr])
    slices = slices.reshape(1,arr.shape[0], arr.shape[1], arr.shape[2], 1)
    slices_cat = np.array([get_category(category)])
    
    return slices, slices_cat


# In[81]:


def get_slices_per_group_3d_all(paths, categories):
    group = None
    group_cat = None

    count = 1
    for i in range(len(paths)):
        path = paths[i]
        try:
            if i == 0:
                group, group_cat = get_slices_3d_all(path, categories[i])
            else:
                new_group, new_group_cat = get_slices_3d_all(path, categories[i])
                group = np.concatenate((group, new_group))
                group_cat = np.concatenate((group_cat, new_group_cat))

            print("-> [%d/%d] Image processed." %(count,len(paths)))
            count+=1
        except:
            print("Error in", path)
    return group, group_cat


# In[82]:


#test with one image
slices, slices_cat = get_slices_3d_all(X_dataset[11], y_dataset[0])
print(slices.shape)
print(slices_cat.shape)

slices, slices_cat = get_slices_per_group_3d_all(X_dataset[:3], y_dataset[:3])
print(slices.shape)
print(slices_cat.shape)

# # MODEL (BINARY)

# ## 3D all brain


X_3d_all_train, y_3d_all_train = get_slices_per_group_3d_all(X_train, y_train)
X_3d_all_test, y_3d_all_test = get_slices_per_group_3d(X_test, y_test)


print("Train:",X_3d_all_train.shape, len(y_3d_all_train))
print("Test:",X_3d_all_test.shape, len(y_3d_all_test))

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv3D, MaxPool3D , Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf


model = Sequential()
model.add(Conv3D(input_shape=(150,180,180,1),filters=20,kernel_size=3,padding="same", activation="relu"))
model.add(Conv3D(filters=64,kernel_size=3,padding="same", activation="relu"))
model.add(Dropout(0.4))
model.add(MaxPool3D(pool_size=2,strides=(2,2,2)))
model.add(Conv3D(filters=128, kernel_size=3, padding="same", activation="relu"))
model.add(Conv3D(filters=128, kernel_size=3, padding="same", activation="relu"))
model.add(Dropout(0.6))
model.add(MaxPool3D(pool_size=2,strides=(2,2,2)))
model.add(Conv3D(filters=256, kernel_size=3, padding="same", activation="relu"))
model.add(Conv3D(filters=256, kernel_size=3, padding="same", activation="relu"))
model.add(Conv3D(filters=256, kernel_size=3, padding="same", activation="relu"))
model.add(Conv3D(filters=256, kernel_size=3, padding="same", activation="relu"))
model.add(Dropout(0.8))
model.add(MaxPool3D(pool_size=2,strides=(2,2,2)))
model.add(Conv3D(filters=512, kernel_size=3, padding="same", activation="relu"))
model.add(Conv3D(filters=512, kernel_size=3, padding="same", activation="relu"))
model.add(Conv3D(filters=512, kernel_size=3, padding="same", activation="relu"))
model.add(Conv3D(filters=512, kernel_size=3, padding="same", activation="relu"))
model.add(Dropout(0.8))
model.add(MaxPool3D(pool_size=2,strides=(2,2,2)))
model.add(Conv3D(filters=512, kernel_size=3, padding="same", activation="relu"))
model.add(Conv3D(filters=512, kernel_size=3, padding="same", activation="relu"))
model.add(Conv3D(filters=512, kernel_size=3, padding="same", activation="relu"))
model.add(Conv3D(filters=512, kernel_size=3, padding="same", activation="relu"))
model.add(Dropout(0.4))
model.add(MaxPool3D(pool_size=2,strides=(2,2,2)))

#Dense layer
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=1000,activation="relu"))
model.add(Dense(1, activation="softmax"))

model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

print(model.summary())


checkpoint = ModelCheckpoint("vgg19_3d_all_14122021.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')

hist = model.fit(X_3d_all_train, y_3d_all_train, batch_size=10,epochs=10,validation_split=0.2, callbacks=[checkpoint])


import matplotlib.pyplot as plt

plt.plot(hist.history["accuracy"])
plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.savefig('3d_graph_13122021.png')

model.load_weights("./vgg19_3d_all_14122021.h5")
test_results = model.evaluate(X_3d_all_test, y_3d_all_test)
test_results