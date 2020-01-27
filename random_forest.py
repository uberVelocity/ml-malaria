import os
import pydot
import numpy as np
import config
import data_wrappers 
from data_wrappers import load_image_data
import pandas as pd
import matplotlib.pyplot as plt
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
    corrects = metrics.accuracy_score(test_labels, predictions, normalize= False)
    print('number of correct:', corrects)
    accuracy = 100 * (corrects / test_labels.shape[0])
    print('Accuracy:', accuracy, '%.')

    #draw_tree(forest, feature_list)
    return accuracy


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
    train_random_forest(train_features, test_features, train_labels, test_labels,feature_list,rf)

    print("###########Smaller tree")
    rf_small = RandomForestClassifier(n_estimators=10, random_state=42)  # Limit depth of tree to 2 levels
    train_random_forest(train_features, test_features, train_labels, test_labels,feature_list, rf_small)

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

    errors = metrics.accuracy_score(test_labels, predictions, normalize = False)

    # Display the performance metrics
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

    accuracy = np.mean(100 * (errors / test_labels.shape[0]))

    print('Accuracy:', round(accuracy, 2), '%.')

    draw_tree(rf_most_important, important_indices)

def rand_forest_n_Fold(): 
    entries = os.environ["HOME"] + config.image_location
    parasitized_entries = os.listdir(entries + '/Parasitized')
    uninfected_entries = os.listdir(entries + '/Uninfected')

    image, features, labels = load_image_data() 
    print('loaded feature shape:',features.shape)
    feature_list = pd.Series(features[0])
    features = np.array(features)

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.5,
                                                                                random_state=42) 
    print('train feature shape:',train_features.shape) 
    print('test feature shape:', test_features.shape) 

    n_best=0
    acc =0
    cost_test=[]
    for n_estimationin in range(10): 
        n = n_estimationin+1
        n *=100
        print("##### n estimation",n) 
        rf = RandomForestClassifier(n_estimators=n, random_state=42, criterion="entropy")
        acc_i = train_random_forest(train_features, test_features, train_labels, test_labels,feature_list,rf)
        tree = rf.estimators_[0]
        cost_test.append(acc_i)
        print('The depth of this tree is:', tree.tree_.max_depth)
        if acc_i > acc:
            n_best = n 
            acc = acc_i 
            rf_best = rf  

    #best setting 
    print("best setting: n =", n_best)
    print(rf_best.base_estimator_)
    draw_tree(rf_best, feature_list) 

    for n_fold in range(10): 
        test_size_n = (n_fold+1)/10 
        print("test size n", test_size_n)
        rf = RandomForestClassifier(n_estimators=n_best, random_state=42, criterion = 'entropy')
        print("##### test size", test_size_n)
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=test_size_n,
                                                                                random_state=42)
        train_random_forest(train_features, test_features, train_labels, test_labels,feature_list,rf) 

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Test ='+str((int(10-split*10)/10.0))+' and Train ='+str(split)+' split, learning rate ='+str(lr))
    ax.plot(cost_test, label ="Test") 
    ax.set_xlabel('n estimator')
    ax.set_ylabel("Cost") 
    ax.legend()
    plt.show()


#rand_forest_one()
rand_forest_n_Fold()
