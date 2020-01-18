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

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(config.scaled_size[1], config.scaled_size[0], 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Flatten the tensors and create Dense layers for classification
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))

    # Output layer
    model.add(layers.Dense(2, activation='softmax'))

    return model


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


if __name__ == '__main__':
    print("Loading images...")
    images, labels = load_images()
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.33, random_state=42)
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
