import os
import numpy as np
import config

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


def feature_scaling(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


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
    k = 2
    images, labels = load_images()
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.33, random_state=42)
    print(f"Images loaded! x_train: f{len(x_train)}, x_test: f{len(x_test)}")
    print(f"(y_train: {len(y_train)}, y_test: {len(y_test)})")

    print("\n Creating classifier...")
    classifier = KNeighborsClassifier(k)

    print("Commencing training...")
    classifier.fit(x_train, y_train)

    print("Predicting...")
    y_pred = classifier.predict(x_test)

    print("Evaluating...")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))