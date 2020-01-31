import matplotlib.pyplot as plt

data = {
    2: [73.0, 1.2],
    3: [74.9, 1.1],
    4: [74.6, 0.8],
    5: [75.0, 0.6],
    6: [75.1, 0.9],
    7: [75.4, 0.8],
    8: [75.2, 0.8],
    9: [75.2, 1.0],
    10: [75.1, 0.6]
}

if __name__ == '__main__':
    # Plot data
    means = [value[0] for value in data.values()]
    stds = [value[1] for value in data.values()]
    plt.errorbar(data.keys(), means, stds, linestyle='None', marker='o')

    # Set axes and title
    plt.title('Mean accuracy per K for KNN')
    plt.xlabel('K')
    plt.ylabel('Accuracy (%)')

    # Save and show figure
    plt.savefig('knn_parametersweep.png')
    plt.show()
