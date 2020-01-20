import os
import numpy as np
import config

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models
from load_images import load_images


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


if __name__ == '__main__':
    print("Loading images...")
    images, labels = load_images()
    # print(f"Images loaded! x_train: f{len(x_train)}, x_test: f{len(x_test)}")
    # print(f"(y_train: {len(y_train)}, y_test: {len(y_test)})")

    n_folds = 10
    BATCH_SIZE = 128
    val_accuracy = [None] * 10
    for i in range(n_folds):
        print('Training on fold ', i + 1)
        print("\n Creating model...")
        model = create_model()
        model.summary()
        
        print("Compiling...")
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("Compiling done! Commencing training...")
        
        x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state = i)
        val_accuracy[i] = model.fit(x_train, y_train, epochs=10, batch_size=BATCH_SIZE,
                            validation_data=(x_test, y_test))
    print('validation accuracies: ', val_accuracy)
    print('mean accuracy: ', mean(val_accuracy))