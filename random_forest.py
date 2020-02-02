import pydot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_wrappers import load_image_data
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel


def train_random_forest(train_features, test_features, train_labels, test_labels, forest):
    forest.fit(train_features, train_labels)
    predictions = forest.predict(test_features)
    corrects = metrics.accuracy_score(test_labels, predictions, normalize=False)
    print('number of correct:', corrects)
    accuracy = (corrects / test_labels.shape[0])
    print('Accuracy:', accuracy)
    tree = forest.estimators_[0]
    print('The depth of this tree is:', tree.tree_.max_depth)

    return accuracy, tree.tree_.max_depth


def draw_tree(rf, feature_list, name):
    # Pull out one tree from the forest
    tree = rf.estimators_[0]
    export_graphviz(tree, out_file=name+'.dot', feature_names=feature_list, rounded=True, precision=1)
    (graph,) = pydot.graph_from_dot_file(name+'.dot')
    graph.write_png(name+'.png')
    print(name+".png created")


def draw_cost(estimations, oob_estimation, acc_estimation):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Test performance')
    ax.plot(estimations, oob_estimation, label="OOB rate")
    ax.plot(estimations, acc_estimation, label="Accuracy")
    ax.set_xlabel('n estimator')
    ax.set_ylabel("Rate of accuracy or OOB") 
    ax.legend()
    plt.show()


def rand_forest_n_fold():
    image, features, labels = load_image_data() 
    print('loaded feature shape:', features.shape)
    
    # Conversions needed for drawing trees
    feature_list = pd.Series(features[0])
    features = np.array(features)

    # Split data
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.5,
                                                                                random_state=42) 
    print('train feature shape:', train_features.shape)
    print('test feature shape:', test_features.shape) 

    n_best = 100
    acc = 0
    depth_list = []
    acc_estimation = []
    oob_estimation = []

    # parameter sweep
    estimations = [10, 100, 150, 300, 500, 1000]
    for estimation in estimations:
        print("##### n estimation", estimation)
        rf = RandomForestClassifier(warm_start=True, n_estimators=estimation, criterion="entropy", oob_score=True)
        acc_i, depth_tree = train_random_forest(train_features, test_features, train_labels, test_labels, rf)
        print("OOB score: ", rf.oob_score_)
        oob_estimation.append(rf.oob_score_)
        acc_estimation.append(acc_i)
        depth_list.append(depth_tree)  
        if acc_i > acc:
            n_best = estimation
            acc = acc_i 

    # best setting
    print("best setting: n =", n_best)
    
    rf = RandomForestClassifier(n_estimators=n_best, random_state=42, criterion="entropy")
    train_random_forest(train_features, test_features, train_labels, test_labels, rf)
    draw_tree(rf, feature_list, "Tree") 
 
    print("###########Important features")
    importances = list(rf.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in zip(range(20), feature_importances)]
    
    sfm = SelectFromModel(rf, threshold=0.04) 
    sfm.fit(train_features, train_labels)

    # reducing number of features in test and train
    train_selected_feature = sfm.transform(train_features)
    test_selected_feature = sfm.transform(test_features)
    print("selected features shapes", train_selected_feature.shape)
    print("selected feature::::", train_selected_feature)

    # training on selected features only
    rf_important = RandomForestClassifier(warm_start=True, n_estimators=n_best, random_state=42, oob_score=True,
                                          criterion="entropy")
    train_random_forest(train_selected_feature, test_selected_feature, train_labels, test_labels, rf_important)
    feature_list = pd.Series(train_selected_feature[0])
    print("RF trained from selected features OOB", rf_important.oob_score_)
    draw_tree(rf_important, feature_list, "selectedTree") 
    draw_cost(estimations, oob_estimation, acc_estimation)


rand_forest_n_fold()
