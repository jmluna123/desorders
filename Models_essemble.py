import numpy as np
import SimpleITK as sitk
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from glob import glob

BUY_file_paths =  sorted(glob("./SEGMENTED/BUY/*.nii.gz"))  #68 files
EAT_file_paths =  sorted(glob("./SEGMENTED/EAT/*.nii.gz"))  #85 files
GAMBLE_file_paths =  sorted(glob("./SEGMENTED/GAMBLE/*.nii.gz")) #7 files
SEX_file_paths =  sorted(glob("./SEGMENTED/SEX/*.nii.gz"))  #42 files
PD_file_paths =  sorted(glob("./SEGMENTED/PD/*.nii.gz"))  #100 files

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


### PREPROSSESING

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


# ## AXIAL (3 channels)
# arr[ xxx , : , : ] axial

import cv2

SLICE_NUMBER = 50
def get_slices_axial_3c(path, category):
  img = sitk.ReadImage(path, sitk.sitkFloat64)
  arr = sitk.GetArrayFromImage(img)

  #normalize the matrix, numbers between 0.0 - 1.0
  arr = arr / arr.max()

  slices = None
  slices_cat = None
  count = 0

  for i in range(arr.shape[0]):
    slice = arr[arr.shape[0] - i -1, : , : ]
    slice = cv2.resize(slice, (224, 224), interpolation=cv2.INTER_CUBIC)
    slice[slice < 0] = 0
    if slice.max() != 0:
        slice = cv2.merge((slice,slice,slice))
        if count == 0:
            slices = np.array([slice])
            slices_cat = np.array([get_category(category)])
        if count < SLICE_NUMBER:
            slices = np.concatenate((slices,[slice]))
            slices_cat = np.concatenate((slices_cat,[get_category(category)]))
        count+=1
  #print("->", count , "slices of", arr.shape[0], "where used for image", path)
  return slices, slices_cat


def get_slices_per_group_axial_3c(paths, categories):
  group = None
  group_cat = None

  count = 1
  for i in range(len(paths)):
    path = paths[i]
    if i == 0:
      group, group_cat = get_slices_axial_3c(path, categories[i])
    else:
      new_group, new_group_cat = get_slices_axial_3c(path, categories[i])
      group = np.concatenate((group, new_group))
      group_cat = np.concatenate((group_cat, new_group_cat))

    print("-> [%d/%d] Slices processed %d." %(count,len(paths), group.shape[0] ))
    count+=1
  return group, group_cat


# In[100]:


#test with one image
i = 3
slices, slices_cat = get_slices_per_group_axial_3c(X_dataset[0:i], y_dataset[0:i])
print(slices.shape)
print(slices_cat.shape)

# # MODEL (BINARY)
# ## Axial (Enssemble)

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


X_axial3c_train, y_axial3c_train = get_slices_per_group_axial_3c(X_train, y_train)


# In[104]:


X_axialec_test, y_axial3c_test = get_slices_per_group_axial_3c(X_test, y_test)


# In[105]:


batch_size = 32
img_height, img_width = 224, 224
input_shape = (img_height, img_width, 3)
epochs = 1000
num_classes = 1


# In[107]:


from tensorflow.keras.applications import VGG19

input_tensor = Input(shape = input_shape)  

base_model =VGG19(input_shape= input_shape,weights='imagenet', include_top=False, input_tensor=input_tensor)

x = base_model.output
x = GlobalAveragePooling2D()(x)

predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=input_tensor,outputs=predictions)


# In[109]:


from keras.applications.densenet import DenseNet169
from keras.applications.inception_v3 import InceptionV3
from keras.applications.nasnet import NASNetMobile
from tensorflow.keras.applications import VGG19

input_tensor = Input(shape = input_shape)  

base_model1=NASNetMobile(input_shape= input_shape,weights='imagenet', include_top=False, input_tensor=input_tensor)
base_model2=InceptionV3(input_shape= input_shape,weights='imagenet', include_top=False, input_tensor=input_tensor)
base_model3=DenseNet201(input_shape= input_shape,weights='imagenet', include_top=False, input_tensor=input_tensor)
base_model4=VGG19(input_shape= input_shape,weights='imagenet', include_top=False, input_tensor=input_tensor)

x1 = base_model1.output
x1 = GlobalAveragePooling2D()(x1)

x2 = base_model2.output
x2 = GlobalAveragePooling2D()(x2)

x3 = base_model3.output
x3 = GlobalAveragePooling2D()(x3)

x4 = base_model4.output
x4 = GlobalAveragePooling2D()(x4)

merge = concatenate([x1, x2, x3 , x4])
predictions = Dense(num_classes, activation='softmax')(merge)

model = Model(inputs=input_tensor,outputs=predictions)


# In[110]:


bottleneck_final_model = Model(inputs=model.input, outputs=merge)


# In[116]:


# training call backs 
lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, epsilon=0.0001, patience=3, verbose=1)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)


# In[118]:


epochs = 1000
dropout_rate = 0.5

model = Sequential()
# model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(1, activation=tf.nn.softmax))

adam_opt2=Adam(lr = 0.0001, beta_1=0.7, beta_2=0.995, amsgrad=True)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss=keras.losses.binary_crossentropy,metrics=['accuracy'])

history = model.fit(X_axial3c_train, y_axial3c_train,
                    epochs=epochs,
                    batch_size=30,
                    callbacks=[lr_reduce, es_callback],
                    validation_split=0.2,
                    verbose= 2)

with open(extracted_features_dir+'history_'+model_name+'.txt','w') as f:
    f.write(str(history.history))