from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from tensorflow.keras import datasets, layers, models

# load 7 things
# append 7 things into one thing
# append class of image and store it as a touple -> (img, class)
# train_test split
# you're done!

# (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# print(type(train_images))
# print(train_images.shape)
# print(len(train_images))
# print(type(train_labels))
# print(train_labels.shape)
# print(len(train_labels))


# image_0 = train_images[0]
# print('First image of train_images')
# print(type(image_0))
# print(image_0.shape)
# print(len(image_0))

# train_images = np.append(train_images, train_images[0])
# print('after append')

# print(type(train_images))
# print(train_images.shape)
# print(len(train_images))

#(img, class)

### 1. Data definition
# Define class labels
labels = []
loadedData = []
dataset = os.environ["HOME"] + "/Documents/University/roboticsAI/recognition/cnn/augmented_dataset/dataset"
folders = os.listdir(dataset)
loaded_images = np.array([])
loaded_labels = np.array([])
old_len = 0
counter = 0
for folder in folders:
    folderName = folder
    folder = os.path.join(dataset, folder)
    if os.path.isdir(folder):
        saveFolder = os.path.join(folder,"Cropped")
        folderAddress = saveFolder + "/" + folderName + '-data.npy'
        labels.append(folderName)
        loaded = np.load(folderAddress)
        if counter == 0:
            loaded_images = loaded
        else:
            loaded_images = np.concatenate((loaded_images, loaded), axis=0)
        for label in range(0, len(loaded)):
            loaded_labels = np.append(loaded_labels, folderName)
        print('length = ', len(loaded))
        print(type(loaded[0][0][0][0]))
        # print(loaded_images.shape)
        # print(loaded_labels.shape)
        counter += 1
        print(counter)
# Load training data
# TODO: APPEND ALL IMAGES (loaded_images[i]) in ndarray that has shape (n, 180, 220, 3) where n is number of total images to append 
print(loaded_images.shape)
print(loaded_labels.shape)

# TODO: Change train_test_split to K-fold cross-validation (k=10) for better generalization.
X_train, X_test, y_train, y_test = train_test_split(loaded_images, loaded_labels, test_size=0.33, random_state=42)

print(len(X_train))
print(len(y_train))

print('---')

print(len(X_test))
print(len(y_test))

### 2. Neural network definition
# Create Convolutional Neural Network with 3 c-layers
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(180, 220, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# # Flatten the tensors and create Dense layers for classification
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

# # 7 Output classes
model.add(layers.Dense(7, activation='softmax'))

model.summary()

# ### 3. Compiling the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ### 4. Training the model
history = model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_test, y_test))

