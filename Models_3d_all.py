import numpy as np
import SimpleITK as sitk
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from glob import glob
from typing import  List

BUY_file_paths =  sorted(glob("./PREPROCESSED/BUY/*.nii.gz"))  #42 files
EAT_file_paths =  sorted(glob("./PREPROCESSED/EAT/*.nii.gz"))  #81 files
GAMBLE_file_paths =  sorted(glob("./PREPROCESSED/GAMBLE/*.nii.gz")) #7 files
SEX_file_paths =  sorted(glob("./PREPROCESSED/SEX/*.nii.gz"))  #22 files

PD_file_paths =  sorted(glob("./PREPROCESSED/PD/*.nii.gz"))  #116 files

print(len(BUY_file_paths), BUY_file_paths[:3])
print(len(EAT_file_paths), EAT_file_paths[:3])
print(len(GAMBLE_file_paths), GAMBLE_file_paths[:3])
print(len(SEX_file_paths), SEX_file_paths[:3])

print(len(PD_file_paths), PD_file_paths[:3])

# ## SPLIT DATA

X_dataset = []
y_dataset = []

X_dataset.extend(PD_file_paths), y_dataset.extend([1] * len(PD_file_paths))
X_dataset.extend(BUY_file_paths), y_dataset.extend([2] * len(BUY_file_paths))
X_dataset.extend(EAT_file_paths), y_dataset.extend([3] * len(EAT_file_paths))
#X_dataset.extend(GAMBLE_file_paths), y_dataset.extend([4] * len(GAMBLE_file_paths))
#X_dataset.extend(SEX_file_paths), y_dataset.extend([5] * len(SEX_file_paths))

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

### 3D All Brain

import cv2

dimension = 180
def get_slices_3d_all(path, category):
    img = sitk.ReadImage(path, sitk.sitkFloat64)
    arr = sitk.GetArrayFromImage(img)
    
    slice = arr[40:120, : , :]
    
    arr = np.zeros([slice.shape[0], dimension,dimension])
    
    count = 0
    for i in range(arr.shape[0]):
        if arr[i:,:].max() != 0.0:
            print(count)
        arr[i, : , :] = cv2.resize(slice[i, : , :], (dimension, dimension), interpolation=cv2.INTER_CUBIC)
        arr[arr<0] = 0
        count += 1
    
    slices = np.array([arr])
    slices = slices.reshape(1,arr.shape[0], arr.shape[1], arr.shape[2], 1)
    slices_cat = np.array([get_category(category)])
    
    return slices, slices_cat

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

#test with one image
slices, slices_cat = get_slices_3d_all(X_dataset[11], y_dataset[0])
print(slices.shape)
print(slices_cat.shape)

slices, slices_cat = get_slices_per_group_3d_all(X_dataset[:3], y_dataset[:3])
print(slices.shape)
print(slices_cat.shape)

# # MODEL (BINARY)

import os
import numpy as np
np.random.seed(777)
import math
import tensorflow as tf
import keras
import keras.backend as K
import h5py
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Activation, merge, Dense, Flatten, Dropout, concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, add, GlobalAveragePooling2D
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score,roc_curve, confusion_matrix, roc_auc_score, auc, f1_score
from keras.regularizers import l2
from keras.applications.xception import Xception, preprocess_input
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications import DenseNet121

from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Lambda,Concatenate
from keras.layers import SeparableConv2D, AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Add

import matplotlib.pyplot as plt

plt.rcParams["axes.grid"] = False
plt.rcParams.update({'font.size': 20})


import keras
from keras.models import Sequential
from keras.layers import Dense, Conv3D, MaxPool3D , Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

# ## 3D all brain

# In[17]:


X_3d_all_train, y_3d_all_train = get_slices_per_group_3d_all(X_train[:50], y_train[:50])


# In[271]:


X_3d_all_test, y_3d_all_test = get_slices_per_group_3d_all(X_test, y_test)


# In[18]:


print("Train:",X_3d_all_train.shape, len(y_3d_all_train))
#print("Test:",X_3d_all_test.shape, len(y_3d_all_test))


# In[19]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Conv3D, MaxPool3D , Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf


# In[20]:


model = Sequential()
model.add(Conv3D(input_shape=(80,180,180,1),filters=20,kernel_size=3,padding="same", activation="relu"))
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


# In[21]:


lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, epsilon=0.0001, patience=3, verbose=1)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)


# In[22]:


checkpoint = ModelCheckpoint("vgg19_3d_all_5012022_1802.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)


# In[ ]:


batch_size = 10
epochs = 1000
dropout_rate = 0.2

history = model.fit(X_3d_all_train, y_3d_all_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[lr_reduce, es_callback],
                    validation_split=0.25,
                    verbose= 2)