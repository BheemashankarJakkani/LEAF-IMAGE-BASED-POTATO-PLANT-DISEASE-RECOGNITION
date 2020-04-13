#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import numpy as np
import pickle
import cv2
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# # Initialising Variables

# In[2]:


default_image_size = tuple((256, 256))
image_size = 0
directory_root = 'E:/Study/Final Year Project/dataset1/'
width=256
height=256
depth=3


# # Function to convert images to array

# In[3]:


def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None


# # Fetch images from directory

# In[4]:


image_list, label_list = [], []
try:
    print("Loading images ...")
    root_dir = listdir(directory_root)
    
    for plant_folder in root_dir :
        plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")
        
        
        for plant_disease_folder in plant_disease_folder_list:
            print(f"Processing {plant_disease_folder} ...")
            plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/")
                
           
            for image in plant_disease_image_list[:200]:
                image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}"
                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                    image_list.append(convert_image_to_array(image_directory))
                    label_list.append(plant_disease_folder)
    print("Image loading completed")  
except Exception as e:
    print(f"Error : {e}")


# # Image Labels uisng LabelBinarizer

# In[5]:


label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)
#pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
n_classes = len(label_binarizer.classes_)


# In[6]:


#Print the classes
print(label_binarizer.classes_)


# In[7]:


print(image_labels[0])
print(n_classes)


# #  Normalize i/p in the range  [0-1]

# In[8]:


np_image_list = np.array(image_list, dtype=np.float16) / 255.0


# In[9]:


np_image_list.shape


# In[10]:


np_image_list.ndim


# In[11]:


np_image_list.min()


# In[12]:


np_image_list.max()


# # Spliting data to train, test data

# In[13]:


print("Spliting data to train, test")
x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state = 0) 


# # Using ImageDataGenerator

# In[14]:


aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")


# # Applying CNN to Dataset

# In[15]:


from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.optimizers import Adam


# In[16]:




model = Sequential()
inputShape = (height, width, depth)
chanDim = -1
if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
    chanDim = 1
model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(n_classes))
model.add(Activation("softmax"))


# In[17]:


model.summary()


# In[18]:


EPOCHS = 25
INIT_LR = 1e-3
BS = 32
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
# train the network
print("...training network...")


# In[19]:


history = model.fit_generator(
    aug.flow(x_train, y_train, batch_size=BS),
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // BS,
    epochs=EPOCHS, verbose=1
    )


# # Plot the train and val curve

# In[20]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()


# # Model Accuracy

# In[21]:


print("[INFO] Calculating model accuracy")
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")


# # Save model using Pickle

# In[22]:


# save the model to disk
print("[INFO] Saving model...")
pickle.dump(model,open('cnn_model.pkl', 'wb'))


# In[43]:


import h5py
model.save('trained_model.h5')


# # Predictions

# In[61]:


#fileobject = open('cnnmodel.pkl', 'rb')
#model = pickle.load(file_object)

imgpath0='E:\\Study\\Final Year Project\\dataset1\\Potato\\Potato___Early_blight\\pearlyblight.JPG'
imgpath1='E:\\Study\\Final Year Project\\dataset1\\Potato\\Potato___Late_blight\\plateblight.JPG'
imgpath2='E:\\Study\\Final Year Project\\dataset1\\Potato\\Potato___healthy\\phealthy.JPG'

imar = convert_image_to_array(imgpath2)
npimagelist = np.array([imar], dtype=np.float16) / 225.0 
PREDICTEDCLASSES2 = model.predict_classes(npimagelist) 
print(PREDICTEDCLASSES2)

#This should print you the category of the prediction.
print (label_binarizer.classes_[PREDICTEDCLASSES2])


# # Predictions from the H5 file

# In[8]:


from keras.models import load_model
import numpy as np
import pickle
import cv2
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array

default_image_size = tuple((256, 256))
image_size = 0
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None
    
model = load_model('trained_model.h5')
imgpath0='E:\\Study\\Final Year Project\\dataset1\\Potato\\Potato___Early_blight\\pearlyblight.JPG'
imgpath1='E:\\Study\\Final Year Project\\dataset1\\Potato\\Potato___Late_blight\\plateblight.JPG'
imgpath2='E:\\Study\\Final Year Project\\dataset1\\Potato\\Potato___healthy\\phealthy.JPG'

imar = convert_image_to_array(imgpath2)
npimagelist = np.array([imar], dtype=np.float16) / 225.0 
PREDICTEDCLASSES2 = model.predict_classes(npimagelist) 
print(PREDICTEDCLASSES2)

#This should print you the category of the prediction.
#print (label_binarizer.classes_[PREDICTEDCLASSES2])
if PREDICTEDCLASSES2==2:
    print("Potato Healthy")
elif PREDICTEDCLASSES2==1:
    print("Potato Late Blight")
else:
    print("Potato Early Blight")

