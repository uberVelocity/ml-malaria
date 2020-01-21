import os
import pydot
import numpy as np
import config
import pandas as pd

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

entries = os.environ["HOME"] + config.image_location
parasitized_entries = os.listdir(entries + '/Parasitized')
uninfected_entries = os.listdir(entries + '/Uninfected')
number_bin = 4  # in image description
number_datasamples = 50

def image_descriptor(im):
    out_hist = []  # historgram
    height, width = im.size
    partial_h = int(height / number_bin)
    partial_w = int(width / number_bin)
    im_new = im
    new_pix = im_new.load()

    for i in range(number_bin):
        for j in range(number_bin):
            r = b = g = counter = 0
            for h in range(i * partial_h, (i + 1) * partial_h):
                for w in range(j * partial_w, (j + 1) * partial_w):
                    # object description here
                    # averaging the color with the window
                    r += new_pix[h, w][0]
                    g += new_pix[h, w][1]
                    b += new_pix[h, w][2]
                    counter += 1
            r /= counter
            g /= counter
            b /= counter
            out_hist.append(r)
            out_hist.append(g)
            out_hist.append(b)
    return out_hist


def create_data_frame():
    out = pd.DataFrame(columns=['label', 'histogram'])
    label = 'parasitized'
    for index, entry in zip(range(int(number_datasamples / 2)), parasitized_entries):
        im = Image.open(entries + '/Parasitized/' + entry)
        hist = image_descriptor(im)
        out.loc[len(out)] = [label, hist]

    label = 'Uninfected'
    for index, entry in zip(range(int(number_datasamples / 2)), uninfected_entries):
        im = Image.open(entries + '/Uninfected/' + entry)
        hist = image_descriptor(im)
        out.loc[len(out)] = [label, hist]

    tags = out['histogram'].apply(pd.Series)
    tags = tags.rename(columns=lambda x: 'hist_' + str(x))
    out = pd.concat([out[:], tags[:]], axis=1)
    out = out.drop(columns=['histogram'], axis=1)
    out = out.fillna(0)
    out.to_csv("malaria.csv")
    return out


def train_random_forest(train_features, test_features, train_labels, test_labels,feature_list,forest):
    forest.fit(train_features, train_labels)
    predictions = forest.predict(test_features)
    errors = metrics.accuracy_score(test_labels, predictions)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels.shape[0])
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')

    draw_tree(forest, feature_list)


def draw_tree(rf, feature_list):
    # Pull out one tree from the forest
    tree = rf.estimators_[0]

    export_graphviz(tree, out_file='tree.dot', feature_names=feature_list, rounded=True, precision=1)
    (graph,) = pydot.graph_from_dot_file('tree.dot')
    graph.write_png('tree.png')

    print('The depth of this tree is:', tree.tree_.max_depth)


def rand_forest_one(): 
    entries = os.environ["HOME"] + config.image_location
    parasitized_entries = os.listdir(entries + '/Parasitized')
    uninfected_entries = os.listdir(entries + '/Uninfected')
    number_bin = 4  # in image description
    number_datasamples = 50

    malaria_data = create_data_frame()

    labels = np.array(malaria_data['label'])
    features = malaria_data.drop(columns=['label'], axis=1)
    feature_list = list(features.columns)
    features = np.array(features)

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.5,
                                                                                random_state=21)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    train_random_forest(rf)

    print("###########Smaller tree")
    rf_small = RandomForestClassifier(n_estimators=10, random_state=42)  # Limit depth of tree to 2 levels
    train_random_forest(rf_small)

    print("###########Important features")
    # Get numerical feature importances
    importances = list(rf.feature_importances_)

    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

    # Print out the feature and importances 
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in zip(range(5), feature_importances)]

    # New random forest with only the two most important variables
    rf_most_important = RandomForestClassifier(n_estimators=10, random_state=42)

    # Extract the two most important features
    important_indices = [feature_list.index('hist_42'), feature_list.index('hist_43')]
    train_important = train_features[:, important_indices]
    test_important = test_features[:, important_indices]

    # Train the random forest
    rf_most_important.fit(train_important, train_labels)

    # Make predictions and determine the error
    predictions = rf_most_important.predict(test_important)

    errors = metrics.accuracy_score(test_labels, predictions)

    # Display the performance metrics
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

    mape = np.mean(100 * (errors / test_labels.shape[0]))
    accuracy = 100 - mape

    print('Accuracy:', round(accuracy, 2), '%.')

    draw_tree(rf_most_important, important_indices)

def rand_forest_n_Fold(): 
    entries = os.environ["HOME"] + config.image_location
    parasitized_entries = os.listdir(entries + '/Parasitized')
    uninfected_entries = os.listdir(entries + '/Uninfected')
    number_bin = 4  # in image description
    number_datasamples = 50

    malaria_data = create_data_frame()

    labels = np.array(malaria_data['label'])
    features = malaria_data.drop(columns=['label'], axis=1)
    feature_list = list(features.columns)
    features = np.array(features)

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.5,
                                                                                random_state=21)
    for n_estimationin in range(10): 
        n = n_estimationin+1
        n *=10
        print("##### n estimation",n) 
        rf = RandomForestClassifier(n_estimators=n, random_state=42)
        train_random_forest(train_features, test_features, train_labels, test_labels,feature_list,rf)
    
    #best setting 
    rf = RandomForestClassifier(n_estimators=50, random_state=42)

    for n_fold in range(10): 
        n = n_fold+1
        test_size_n = n/number_datasamples
        print("##### test size", test_size_n)
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=n,
                                                                                random_state=21)
        train_random_forest(train_features, test_features, train_labels, test_labels,feature_list,rf)
    

rand_forest_n_Fold()
