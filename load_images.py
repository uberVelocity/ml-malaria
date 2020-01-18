import config
import os
import numpy as np


def load_images():
    """
    Loads processed images from the .npy file at the configured location.
    """
    dataset_path = os.environ["HOME"] + config.image_location
    folders = os.listdir(dataset_path)

    image_arr = np.array([])
    label_arr = np.array([])
    class_label = 0

    for folder in folders:
        folder_name = folder
        folder = os.path.join(dataset_path, folder)

        if os.path.isdir(folder):
            save_folder = os.path.join(folder, 'augmented')
            folder_address = save_folder + '/' + folder_name + '-data.npy'
            loaded = np.load(folder_address)

            if class_label == 0:
                image_arr = loaded
            else:
                image_arr = np.concatenate((image_arr, loaded), axis=0)

            for label in range(0, len(loaded)):
                label_arr = np.append(label_arr, class_label)

            print('loaded ', len(loaded))
            class_label += 1
            print(class_label)

    return image_arr, label_arr
