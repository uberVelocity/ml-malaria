import cv2
import os
import numpy as np
import config
import imutils

"""
Collects images from folders and produces normalized dataset as .npy files.
"""


def load_image_data():
    # Wrapper for loading data
    raw_images = np.load('raw_images.npz')
    features = np.load('features.npz')
    labels = np.load('labels.npz')

    return raw_images, features, labels


def image_to_feature_vector(image, size=(32, 32)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size).flatten()


def extract_color_histogram(image, bins=(8, 8, 8)):
    # extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])

    # handle normalizing the histogram if we are using OpenCV 2.4.X
    if imutils.is_cv2():
        hist = cv2.normalize(hist, None)

    # otherwise, perform "in place" normalization in OpenCV 3 (I
    # personally hate the way this is done
    else:
        cv2.normalize(hist, hist)

    # return the flattened histogram as the feature vector
    return hist.flatten()


def extract_features():
    dataset_path = os.environ["HOME"] + config.image_location
    folders = os.listdir(dataset_path)

    raw_images = []
    features = []
    labels = []

    for folder in folders:
        folder_path = os.path.join(dataset_path, folder)
        label = folder.lower()  # One class label per directory
        print(f"Working on {label}")

        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)

            for file in files:
                file_path = os.path.join(folder_path, file)

                if os.path.isfile(file_path) and os.path.splitext(file)[-1].lower() == ".png":  # If image file
                    # Create features
                    image = cv2.imread(file_path)
                    pixels = image_to_feature_vector(image)
                    hist = extract_color_histogram(image)

                    # Store in data collection
                    raw_images.append(pixels)
                    features.append(hist)
                    labels.append(label)

    # Convert arrays for easier usage
    raw_images = np.array(raw_images)
    features = np.array(features)
    labels = np.array(labels)

    np.savez_compressed('raw_images.npz', raw_images)
    np.savez_compressed('features.npz', features)
    np.savez_compressed('labels.npz', labels)
    print('Data stored in .npz files')
