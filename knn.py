import numpy as np

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from data_wrappers import load_image_data
from sklearn.metrics import confusion_matrix


def ten_folds_knn(k):
    """
    Performs 10_folds cross validation on KNN using the preprocessed dataset.
    :param k: K hyper-parameter for number of neighbors considered in classification.
    :return: Tuple containing mean and standard deviation
    """
    _, features, labels = load_image_data()  # Images are not used, only features of images
    kf = KFold(n_splits=10, shuffle=True)  # Set values for K-Fold
    k_folds_iterator = 1
    scores = []
    conv_matrix = np.array([[0, 0], [0, 0]])

    # For each fold selected as test fold
    for train_index, test_index in kf.split(labels):
        print(f"{k}-nn iteration {k_folds_iterator}")
        k_folds_iterator += 1

        # Get fold data
        train_features, test_features = features[train_index], features[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        # Training classifier
        knn = KNeighborsClassifier(k)
        knn.fit(train_features, train_labels)

        # Testing
        predictions = knn.predict(test_features)
        score = knn.score(test_features, test_labels)
        scores.append(score)
        conv_matrix += confusion_matrix(test_labels, predictions)

    print("Cross validation done, results:")
    print(f"Mean accuracy: {round(np.mean(scores)*100, 1)}, std: {round(np.std(scores)*100, 1)}")
    conv_matrix = conv_matrix / 10.0
    tn = conv_matrix[0][0]
    fn = conv_matrix[1][0]
    tp = conv_matrix[1][1]
    fp = conv_matrix[0][1]
    specificity = tn/(tn+fn)
    sensitivity = tp/(tp+fp)

    print(f"True negative: {tn}, \nFalse negative: {fn}, \nTrue positive: {tp}, \nFalse positives: {fp}")
    print(f"Sensitivity: {sensitivity}, specificity: {specificity}")

    return round(np.mean(scores)*100, 1), round(np.std(scores)*100, 1)


def find_optimal_k():
    """
    Function that finds optimal value of K in range [2, ..., 10]
    :return: K values of highest accuracy KNN results
    """
    accuracies = []
    stds = []

    # For k in [2, ..., 10]
    for k in range(9):
        accuracy, std = ten_folds_knn(k + 2)
        accuracies.append(accuracy)
        stds.append(std)

    # Show results
    print(f"Accuracies: {accuracies}")
    print(f"Standard Deviations: {stds}")
    best = int(np.where(accuracies == np.amax(accuracies))[0])
    print(f"Best K = {best + 2} with accuracy {accuracies[best]} ({stds[best]})")

    return best+2


if __name__ == '__main__':
    ten_folds_knn(7)
