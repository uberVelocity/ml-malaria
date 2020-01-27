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
    print(f"Mean accuracy: {round(np.mean(scores)*100, 1)}, std: {round(np.std(scores)*100, 1)}")

    return round(np.mean(scores)*100, 1), round(np.std(scores)*100, 1)


if __name__ == '__main__':
    accuracies = []
    stds = []
    for k in range(9):
        accuracy, std = ten_folds_knn(k+2)
        accuracies.append(accuracy)
        stds.append(std)

    # Show results
    print(f"Accuracies: {accuracies}")
    print(f"Standard Deviations: {stds}")
    best = int(np.where(accuracies == np.amax(accuracies))[0])
    print(f"Best K = {best + 2} with accuracy {accuracies[best]} ({stds[best]})")

