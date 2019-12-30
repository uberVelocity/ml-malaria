import os
import numpy as np
import config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models


def create_model():
    """
    Creates CNN with three convolutional layers and two output classes
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(config[1], config[0], 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Flatten the tensors and create Dense layers for classification
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))

    model.add(layers.Dense(2, activation='softmax'))

    return model


def create_train_and_test():
    """
    Loads processed images from the dataset location and generates train and test sets.
    """
    # Load data from .npy files
    dataset = os.environ["HOME"] + config.image_location
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
                loaded_images = np.concatenate((loaded_images, loaded), axis=0)
            for label in range(0, len(loaded)):
                loaded_labels = np.append(loaded_labels, counter)
            print('loaded ', len(loaded))
            counter += 1
            print(counter)

    # Sanity check
    print(loaded_images.shape)
    print(loaded_labels)
    print(label_dict)

    return train_test_split(loaded_images, loaded_labels, test_size=0.33, random_state=42)


if __name__ == '__main__':
    print("Loading images...")
    x_train, x_test, y_train, y_test = create_train_and_test()
    print(f"Images loaded! x_train: f{len(x_train)}, x_test: f{len(x_test)}")
    print(f"(y_train: {len(y_train)}, y_test: {len(y_test)})")

    print("\n Creating model...")
    model = create_model()
    model.summary()

    print("Compiling...")
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("Compiling done! Commencing training...")
    history = model.fit(x_train, y_train, epochs=10,
                        validation_data=(x_test, y_test))
