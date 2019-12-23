import os
import numpy as np
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models

# Load data from .npy files
dataset = os.environ["HOME"] + "/Documents/University/ml/cell-images-for-detecting-malaria/cell_images/cell_images"
folders = os.listdir(dataset)
labels = []
loaded_images = np.array([])
loaded_labels = np.array([])
counter = 0
label_dict = dict()
for folder in folders:
    folder_name = folder
    label_dict[counter] = folder_name
    folder = os.path.join(dataset, folder)
    if os.path.isdir(folder):
        save_folder = os.path.join(folder, 'augmented')
        folder_address = save_folder + '/' + folder_name + '-data.npy'
        labels.append(counter)
        loaded = np.load(folder_address)
        if counter == 0:
            loaded_images = loaded
        else:
            loaded_images = np.concatenate((loaded_images, loaded), axis = 0)
        for label in range(0, len(loaded)):
            loaded_labels = np.append(loaded_labels, counter)
        print('loaded ', len(loaded))
        counter += 1
        print(counter)

# Sanity check
print(loaded_images.shape)
print(loaded_labels)
print(label_dict)
X_train, X_test, y_train, y_test = train_test_split(loaded_images, loaded_labels, test_size=0.33, random_state=42)

print(len(X_train))
print(len(y_train))

print('---')

print(len(X_test))
print(len(y_test))

### 2. Neural network definition
# Create Convolutional Neural Network with 3 c-layers
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 80, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# # Flatten the tensors and create Dense layers for classification
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

# # 7 Output classes
model.add(layers.Dense(2, activation='softmax'))

model.summary()

# ### 3. Compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ### 4. Training the model
history = model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_test, y_test))

