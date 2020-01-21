import numpy as np

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from data_wrappers import load_image_data


def ten_folds_knn(k):
    _, features, labels = load_image_data()  # Images are not used, only features of images
    kf = KFold(n_splits=10, shuffle=True)  # Set values for K-Fold
    k_folds_iterator = 1
    scores = []
    # For each fold selected as test fold
    for train_index, test_index in kf.split(labels):
        print(f"{k} folds iteration {k_folds_iterator}")
        k_folds_iterator += 1

        # Get fold data
        train_features, test_features = features[train_index], features[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        # Training classifier
        knn = KNeighborsClassifier(k)
        knn.fit(train_features, train_labels)

        # Testing
        score = knn.score(test_features, test_labels)
        scores.append(score)
    print("Cross validation done, results:")
    print(f"Mean accuracy: {np.mean(scores)}, std: {np.std(scores)}")

    return np.mean(scores), np.std(scores)


results = [ten_folds_knn(k+2) for k in range(8)]
print(results)
print(f"Best K = {np.where(results == np.amax(results)) + 2}")

