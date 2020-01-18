from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from load_images import load_images


def feature_scaling(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


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