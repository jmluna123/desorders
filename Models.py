#!/usr/bin/env python
# coding: utf-8


# In[69]:


import numpy as np
import SimpleITK as sitk


# #DATA
# 

# ###GET DATA

# In[70]:


from glob import glob
from typing import  List


# In[71]:


BUY_file_paths=  sorted(glob("./BUY/*.nii.gz"))  #68 files
#EAT_file_paths: List[str] =  sorted(glob("/content/drive/MyDrive/Integradora/Data/EAT/*.nii.gz"))  #files
GAMBLE_file_paths =  sorted(glob("./GAMBLE/*.nii.gz"))  #7 files
SEX_file_paths =  sorted(glob("./SEX/*.nii.gz"))  #42 files

PD_file_paths=  sorted(glob("./PD/*.nii.gz"))  #100 files


# In[72]:


print(len(BUY_file_paths), BUY_file_paths[:3])
#print(len(EAT_file_paths), EAT_file_paths[:3])
print(len(GAMBLE_file_paths), GAMBLE_file_paths[:3])
print(len(SEX_file_paths), SEX_file_paths[:3])

print(len(PD_file_paths), PD_file_paths[:3])



# In[74]:


X_dataset = []
y_dataset = []

X_dataset.extend(PD_file_paths)
y_dataset.extend([1] * len(PD_file_paths))
X_dataset.extend(BUY_file_paths)
y_dataset.extend([2] * len(BUY_file_paths))
X_dataset.extend(GAMBLE_file_paths)
y_dataset.extend([4] * len(GAMBLE_file_paths))
X_dataset.extend(SEX_file_paths)
y_dataset.extend([5] * len(SEX_file_paths))

print("Dataset:", len(X_dataset), len(y_dataset))
# In[75]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, test_size = 0.2, stratify=y_dataset)


# In[76]:

print("Train:", len(X_train), len(y_train))
print("Test:", len(X_test), len(y_test))


# ##PREPROSSESING

# In[8]:


def get_category(category, is_binary = True):
  result = np.zeros(2 if is_binary else 5)
  if is_binary:
    return 1 if category == 1 else 0
  elif category == 1:
    result[0] = 1
  else:
    result[category-1] = 1
  return result


# In[9]:


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


# In[10]:


print(get_category(5, True), get_category(5, False))
print(get_category(1, True), get_category(1, False))


# ## 3D

# In[46]:


def get_slices_3d(path, category):
    img = sitk.ReadImage(path, sitk.sitkFloat64)
    arr = sitk.GetArrayFromImage(img)

    #normalize the matrix, numbers between 0.0 - 1.0
    arr = arr / arr.max()
    
    slices = np.array([arr])
    slices = slices.reshape(1, -1, slices.shape[1], slices.shape[2], 1)
    slices_cat = np.array([get_category(category)])
    
    return slices, slices_cat


# In[47]:

import cv2;
def get_slices_per_group_3d(paths, categories):
    group = None
    group_cat = None

    count = 1
    for i in range(len(paths)):
        path = paths[i]
        if i == 0:
            group, group_cat = get_slices_3d(path, categories[i])
        else:
            new_group, new_group_cat = get_slices_3d(path, categories[i])
            group = np.concatenate((group, new_group))
            group_cat = np.concatenate((group_cat, new_group_cat))

        print("-> [%d/%d] Image processed." %(count,len(paths)))
        count+=1
    return group, group_cat


# In[48]:


#test with one image
slices, slices_cat = get_slices_3d(X_dataset[0], y_dataset[0])
print(slices.shape)
print(slices_cat.shape)


# In[49]:


slices, slices_cat = get_slices_per_group_3d(X_dataset[:3], y_dataset[:3])
print(slices.shape)
print(slices_cat.shape)


# ## AXIAL
# arr[ xxx , : , : ] axial

# In[18]:


def get_slices_axial(path, category):
  img = sitk.ReadImage(path, sitk.sitkFloat64)
  arr = sitk.GetArrayFromImage(img)

  #normalize the matrix, numbers between 0.0 - 1.0
  arr = arr / arr.max()

  slices = None
  slices_cat = None
  count = 0

  for i in range(arr.shape[0]):
    slice = arr[i, : , : ]
    slice = cv2.resize(slice, (224, 224), interpolation=cv2.INTER_CUBIC)
    slice[slice < 0] = 0
    if slice.max() != 0:
      if count == 0:
        slices = np.array([slice])
        slices_cat = np.array([get_category(category)])
      else:
        slices = np.concatenate((slices,[slice]))
        slices_cat = np.concatenate((slices_cat,[get_category(category)]))
      count+=1
  #print("->", count , "slices of", arr.shape[0], "where used for image", path)
  return slices, slices_cat


# In[19]:


def get_slices_per_group_axial(paths, categories):
  group = None
  group_cat = None

  count = 1
  for i in range(len(paths)):
    path = paths[i]
    if i == 0:
      group, group_cat = get_slices_axial(path, categories[i])
    else:
      new_group, new_group_cat = get_slices_axial(path, categories[i])
      group = np.concatenate((group, new_group))
      group_cat = np.concatenate((group_cat, new_group_cat))

    print("-> [%d/%d] Slices processed %d." %(count,len(paths), group.shape[0] ))
    count+=1
  return group, group_cat


# In[20]:


#test with one image
slices, slices_cat = get_slices_axial(X_dataset[0], y_dataset[0])
print(slices.shape)
print(slices_cat.shape)


# ## CORONAL
# arr[ : , xxx , : ] coronal 

# In[21]:


# type: the number of the group
# group_n: the count of diferent groups
# for example,we can have the groups 
#    1) [EAT, GAMBLE, SEX, BUY, PURE] and the image is from GAMBLE
#     -> type = 2
#     -> group_n = 5
#    2) or [PURE, ICD] and the image is from PURE
#     -> type = 1
#     -> group_n = 2
#output:
#     -> normalized array of the slices 
#     -> the category of each image in one-hot encoded.
# for example
#   1) [0,1,0,0,0], and 
#   2) [1,0]
def get_slices_coronal(path, category):
  img = sitk.ReadImage(path, sitk.sitkFloat64)
  arr = sitk.GetArrayFromImage(img)

  #normalize the matrix, numbers between 0.0 - 1.0
  arr = arr / arr.max()

  slices = None
  slices_cat = None
  count = 0

  for i in range(arr.shape[0]):
    slice = arr[:, i , : ]
    slice = cv2.resize(slice, (224, 224), interpolation=cv2.INTER_CUBIC)
    slice[slice < 0] = 0
    if slice.max() != 0:
      if count == 0:
        slices = np.array([slice])
        slices_cat = np.array([get_category(category)])
      else:
        slices = np.concatenate((slices,[slice]))
        slices_cat = np.concatenate((slices_cat,[get_category(category)]))
      count+=1
  #print("->", count , "slices of", arr.shape[0], "where used for image", path)
  return slices, slices_cat


# In[22]:


def get_slices_per_group_coronal(paths, categories):
  group = None
  group_cat = None

  count = 1
  for i in range(len(paths)):
    path = paths[i]
    if i == 0:
      group, group_cat = get_slices_coronal(path, categories[i])
    else:
      new_group, new_group_cat = get_slices_coronal(path, categories[i])
      group = np.concatenate((group, new_group))
      group_cat = np.concatenate((group_cat, new_group_cat))

    print("-> [%d/%d] Slices processed %d." %(count,len(paths), group.shape[0] ))
    count+=1
  return group, group_cat


# In[23]:


#test with one image
slices, slices_cat = get_slices_coronal(X_dataset[0], y_dataset[0])
print(slices.shape)
print(slices_cat.shape)


# ## SAGITAL
# arr[ : , : , xxx ] sagital 

# In[24]:


# type: the number of the group
# group_n: the count of diferent groups
# for example,we can have the groups 
#    1) [EAT, GAMBLE, SEX, BUY, PURE] and the image is from GAMBLE
#     -> type = 2
#     -> group_n = 5
#    2) or [PURE, ICD] and the image is from PURE
#     -> type = 1
#     -> group_n = 2
#output:
#     -> normalized array of the slices 
#     -> the category of each image in one-hot encoded.
# for example
#   1) [0,1,0,0,0], and 
#   2) [1,0]
def get_slices_sagital(path, category):
  img = sitk.ReadImage(path, sitk.sitkFloat64)
  arr = sitk.GetArrayFromImage(img)

  #normalize the matrix, numbers between 0.0 - 1.0
  arr = arr / arr.max()

  slices = None
  slices_cat = None
  count = 0

  for i in range(arr.shape[0]):
    slice = arr[:, : , i ]
    slice = cv2.resize(slice, (224, 224), interpolation=cv2.INTER_CUBIC)
    slice[slice < 0] = 0
    if slice.max() != 0:
      if count == 0:
        slices = np.array([slice])
        slices_cat = np.array([get_category(category)])
      else:
        slices = np.concatenate((slices,[slice]))
        slices_cat = np.concatenate((slices_cat,[get_category(category)]))
      count+=1
  #print("->", count , "slices of", arr.shape[0], "where used for image", path)
  return slices, slices_cat


# In[25]:


def get_slices_per_group_sagital(paths, categories):
  group = None
  group_cat = None

  count = 1
  for i in range(len(paths)):
    path = paths[i]
    if i == 0:
      group, group_cat = get_slices_sagital(path, categories[i])
    else:
      new_group, new_group_cat = get_slices_sagital(path, categories[i])
      group = np.concatenate((group, new_group))
      group_cat = np.concatenate((group_cat, new_group_cat))

    print("-> [%d/%d] Slices processed %d." %(count,len(paths), group.shape[0] ))
    count+=1
  return group, group_cat


# In[26]:


#test with one image
slices, slices_cat = get_slices_sagital(X_dataset[0], y_dataset[0])
print(slices.shape)
print(slices_cat.shape)


# # MODEL (BINARY)

# In[61]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.layers import Conv3D, MaxPool3D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping

# ## AXIAL

# In[15]:

"""
X_axial_train, y_axial_train = get_slices_per_group_axial(X_train, y_train)
X_axial_train = X_axial_train.reshape(-1, X_axial_train.shape[1], X_axial_train.shape[2], 1)

X_axial_test, y_axial_test = get_slices_per_group_axial(X_test, y_test)
X_axial_test = X_axial_test.reshape(-1, X_axial_train.shape[1], X_axial_train.shape[2], 1)


# In[16]:


print("Train: X:(%d, %d, %d, %d), y: %d" %(X_axial_train.shape[0],X_axial_train.shape[1],X_axial_train.shape[2],X_axial_train.shape[3], len(y_axial_train)))
print("Test: X:(%d, %d, %d, %d), y: %d" %(X_axial_test.shape[0],X_axial_test.shape[1],X_axial_test.shape[2],X_axial_test.shape[3], len(y_axial_test)))


# ###VGG19

# In[18]:


model = Sequential()
model.add(Conv2D(input_shape=(224,224,1),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Dropout(0.4))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Dropout(0.6))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Dropout(0.8))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Dropout(0.8))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Dropout(0.4))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

#Dense layer
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=1000,activation="relu"))
model.add(Dense(1, activation="softmax"))

model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

print(model.summary())


# In[19]:


checkpoint = ModelCheckpoint("vgg19_axial.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
#fit_generator(steps_per_epoch=1,generator=traindata, validation_data= testdata, validation_steps=1,epochs=50,callbacks=[checkpoint])
# hist = model.fit(traindata, testdata, batch_size=10, epochs=20, verbose=0, shuffle=True,validation_split=0.2,callbacks=[checkpoint])


# In[21]:


hist = model.fit(X_axial_train, y_axial_train, batch_size=10,epochs=30,validation_split=0.2, callbacks=[checkpoint])



# In[25]:


test_results = model.evaluate(X_axial_test, y_axial_test)
test_results
"""

# # CORONAL

# In[17]:

"""
X_coronal_train, y_coronal_train = get_slices_per_group_coronal(X_train, y_train)
X_coronal_train = X_coronal_train.reshape(-1, X_coronal_train.shape[1], X_coronal_train.shape[2], 1)


# In[18]:


X_coronal_test, y_coronal_test = get_slices_per_group_coronal(X_test, y_test)
X_coronal_test = X_coronal_test.reshape(-1, X_coronal_train.shape[1], X_coronal_train.shape[2], 1)


# In[19]:


print("Train: X:(%d, %d, %d, %d), y: %d" %(X_coronal_train.shape[0],X_coronal_train.shape[1],X_coronal_train.shape[2],X_coronal_train.shape[3], len(y_coronal_train)))
print("Test: X:(%d, %d, %d, %d), y: %d" %(X_coronal_test.shape[0],X_coronal_test.shape[1],X_coronal_test.shape[2],X_coronal_test.shape[3], len(y_coronal_test)))


# # VGG19

# In[21]:


model = Sequential()
model.add(Conv2D(input_shape=(224,224,1),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Dropout(0.4))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Dropout(0.6))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Dropout(0.8))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Dropout(0.8))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Dropout(0.4))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

#Dense layer
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=1000,activation="relu"))
model.add(Dense(1, activation="softmax"))

model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

print(model.summary())


# In[22]:


checkpoint = ModelCheckpoint("vgg19_coronal.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
#fit_generator(steps_per_epoch=1,generator=traindata, validation_data= testdata, validation_steps=1,epochs=50,callbacks=[checkpoint])
# hist = model.fit(traindata, testdata, batch_size=10, epochs=20, verbose=0, shuffle=True,validation_split=0.2,callbacks=[checkpoint])


# In[ ]:


hist = model.fit(X_coronal_train, y_coronal_train, batch_size=10,epochs=2,validation_split=0.2, callbacks=[checkpoint])

test_results = model.evaluate(X_coronal_test, y_coronal_test)
test_results

# # Sagital

# In[ ]:

"""
X_sagital_train, y_sagital_train = get_slices_per_group_sagital(X_train, y_train)
X_sagital_train = X_sagital_train.reshape(-1, X_sagital_train.shape[1], X_sagital_train.shape[2], 1)


# In[ ]:


print("Train: X:(%d, %d, %d, %d), y: %d" %(X_sagital_train.shape[0],X_sagital_train.shape[1],X_sagital_train.shape[2],X_sagital_train.shape[3], len(y_sagital_train)))
print("Test: X:(%d, %d, %d, %d), y: %d" %(X_sagital_test.shape[0],X_sagital_test.shape[1],X_sagital_test.shape[2],X_sagital_test.shape[3], len(y_sagital_test)))

# # VGG19

# In[21]:


model = Sequential()
model.add(Conv2D(input_shape=(224,224,1),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Dropout(0.4))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Dropout(0.6))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Dropout(0.8))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Dropout(0.8))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Dropout(0.4))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

#Dense layer
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=1000,activation="relu"))
model.add(Dense(1, activation="softmax"))

model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

print(model.summary())


# In[22]:


checkpoint = ModelCheckpoint("vgg19_sagital.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
#fit_generator(steps_per_epoch=1,generator=traindata, validation_data= testdata, validation_steps=1,epochs=50,callbacks=[checkpoint])
# hist = model.fit(traindata, testdata, batch_size=10, epochs=20, verbose=0, shuffle=True,validation_split=0.2,callbacks=[checkpoint])


# In[ ]:


hist = model.fit(X_sagital_train, y_sagital_train, batch_size=10,epochs=2,validation_split=0.2, callbacks=[checkpoint])

test_results = model.evaluate(X_sagital_test, y_sagital_test)
test_results

