import numpy as np
import config
import data_wrappers
import matplotlib.pyplot as plt
import sklearn.metrics as skmet

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models


def create_model():
    """
    Creates CNN with three convolutional layers and two output classes
    """
    new_model = models.Sequential()

    # Convolutional layers
    new_model.add(layers.Conv2D(32, (3, 3), activation='relu',
                                input_shape=(config.scaled_size[1], config.scaled_size[0], 3)))
    new_model.add(layers.MaxPooling2D(2, 2))
    new_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    new_model.add(layers.MaxPooling2D(2, 2))
    new_model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Flatten the tensors and create Dense layers for classification
    new_model.add(layers.Flatten())
    new_model.add(layers.Dense(64, activation='relu'))

    # Output layer
    new_model.add(layers.Dense(2, activation='softmax'))

    return new_model


def plot_val_accuracies(val_accuracies):
    plt.plot(val_accuracies, '-o')
    plt.ylabel('Validation accuracy')
    plt.xlabel('Epoch')
    plt.title('CNN Validation accuracy across 10 epochs')
    plt.show()


if __name__ == '__main__':
    # Load images
    print('Loading images...')
    features, temp, labels = data_wrappers.load_image_data()

    # Check images dimensions
    features = np.reshape(features, (27558, 32, 32, 3))
    print(features.shape)

    # Check correct int labels (tensorflow does not accept strings as classes)
    if 'parasitized' in labels:
        labels[labels == 'parasitized'] = 0
    if 'uninfected' in labels:
        labels[labels == 'uninfected'] = 1

    # Convert strings to ints
    labels = list(map(int, labels))
    labels = np.array(labels)
    print(labels.shape)

    # Create CNN model
    print('\n Creating model...')
    model = create_model()
    model.summary()
    print(tf.keras.metrics)

    # Compile model
    print('Compiling...')

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print('Compiling done! Commencing training...')
    
    # Fit the model
    BATCH_SIZE = 128
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=1/3, random_state=42)
    val_accuracy = model.fit(x_train, y_train, epochs=10,
                                validation_data=(x_test, y_test))
    
    # Sample the data once more for good measure
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=1/3, random_state=10)
    
    # Predict data using trained classifier
    y_pred = model.predict(x_test)

    # Convert predictons to binary class output
    y_pred_binary = [None] * len(y_pred)
    for i in range(0, len(y_pred)):
        print(y_pred[i][0], " : ", y_pred[i][1])
        if y_pred[i][0] > y_pred[i][1]:
            y_pred_binary[i] = 0
        else:
            y_pred_binary[i] = 1
    
    # Extract tn, fp, fn, tp
    tn, fp, fn, tp = skmet.confusion_matrix(y_test, y_pred_binary).ravel();
    print(tp, fp, tn, fn)

    # Compute sensitivity and specificity
    sensitivity = tp / (tp + fp)
    specificity = tn / (tn + fn)
    print("sensitivity: ", sensitivity)
    print("specificity: ", specificity)

    # In order to create the plot using this function, you should have Tensorflow version 2.1.0
    # Check tensorflow version by doing `import tensorflow as tf; print(tf.__version__);
    # Print validation accuracy scores - commented by default.
    # print('val_accuracy: ', val_accuracy.history['val_accuracy'])

    # Plot validation accuracy scores - Commented by default.
    # plot_val_accuracies(val_accuracy.history['val_accuracy'])

