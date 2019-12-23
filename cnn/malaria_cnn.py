import os
import numpy as np
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from tensorflow.keras import datasets, layers, models
dataset = os.environ["HOME"] + "/Documents/University/ml/cell-images-for-detecting-malaria/cell_images/cell_images"
folders = os.listdir(dataset)
labels = []
loaded_images = np.array([])
loaded_labels = np.array([])
counter = 0
for folder in folders:
    folder_name = folder
    folder = os.path.join(dataset, folder)
    if os.path.isdir(folder):
        save_folder = os.path.join(folder, 'augmented')
        folder_address = save_folder + '/' + folder_name + '-data.npy'
        labels.append(folder_name)
        loaded = np.load(folder_address)
        if counter == 0:
            loaded_images = loaded
        else:
            loaded_images = np.concatenate((loaded_images, loaded), axis = 0)
        for label in range(0, len(loaded)):
            loaded_labels = np.append(loaded_labels, folder_name)
        print('loaded ', len(loaded))
        counter += 1
        print(counter)

print(loaded_images.shape)
print(loaded_labels.shape)