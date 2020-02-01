import numpy as np
import config
import data_wrappers
import tensorflow as tf
import matplotlib.pyplot as plt

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
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Flatten the tensors and create Dense layers for classification
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))

    # Output layer
    model.add(layers.Dense(2, activation='softmax'))

    return model

def plot_val_accuracies(val_accuracies):
    plt.plot(val_accuracies, '-o')
    plt.ylabel('Validation accuracy')
    plt.xlabel('Epoch')
    plt.title('CNN Validation accuracy across 10 epochs')
    plt.show()

if __name__ == '__main__':
    # Load images
    print('Loading images...')
    features, labels = data_wrappers.load_images()

    # Check images dimensions
    print(features.shape)
    print(labels.shape)

    # Create CNN model
    print('\n Creating model...')
    model = create_model()
    model.summary()

    # Compile model
    print('Compiling...')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print('Compiling done! Commencing training...')
    
    # Fit the model
    BATCH_SIZE = 128
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=1/3, random_state=42)
    val_accuracy = model.fit(x_train, y_train, epochs=3,
                                validation_data=(x_test, y_test))

    # Print validation accuracy scores
    print('val_accuracy: ', val_accuracy.history['val_accuracy'])

    # Plot validation accuracy scores
    plot_val_accuracies(val_accuracy.history['val_accuracy'])

