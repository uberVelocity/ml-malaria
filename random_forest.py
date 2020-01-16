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
parasitized_entries = os.listdir(entries+'/Parasitized')
uninfected_entries = os.listdir(entries+'/Uninfected') 
# global vairable
number_bin = 4 # in image description 
number_datasamples= 50  


def imageDescriptor(im):
    out_hist = [] #historgram
    height, width =im.size
    partial_h = int(height/number_bin) 
    partial_w = int(width/number_bin) 
    im_new = im
    new_pix=im_new.load()
    for i in range(number_bin): 
        for j in range(number_bin): 
            r=b= g= counter =0 
            for h in range(i*partial_h, (i+1)*partial_h): 
                for w in range(j*partial_w, (j+1)*partial_w): 
                    #object description here 
                    ## averaging the color with the window
                    r += new_pix[h,w][0] 
                    g += new_pix[h,w][1]
                    b += new_pix[h,w][2]
                    counter+=1
            r/=counter 
            g/=counter 
            b/=counter 
            out_hist.append(r)
            out_hist.append(g)
            out_hist.append(b)
    # out_hist+=list(des)
    #print(out_hist)
    return out_hist


def createDataFrame(): 
    out = pd.DataFrame(columns=['label','histogram'])
    label = 'parasitized' 
    for index, entry in zip(range(int(number_datasamples/2)),parasitized_entries): 
        im = Image.open(os.getcwd()+'/cell_images/Parasitized/'+entry)
        hist = imageDescriptor(im)
        out.loc[len(out)] = [label,hist] 

    label= 'Uninfected'
    for index, entry in zip(range(int(number_datasamples/2)), uninfected_entries): 
        im = Image.open(os.getcwd()+'/cell_images/Uninfected/'+entry)
        hist = imageDescriptor(im)
        out.loc[len(out)] = [label,hist] 

    tags = out['histogram'].apply(pd.Series)
    tags = tags.rename(columns = lambda x : 'hist_' + str(x)) 
    out = pd.concat([out[:], tags[:]], axis=1)
    out = out.drop(columns=['histogram'], axis=1)
    out = out.fillna(0)
    out.to_csv("malaria.csv")
    return out


malaria_data = createDataFrame() 
print(malaria_data)

labels = np.array(malaria_data['label'])
features= malaria_data.drop(columns=['label'], axis = 1)
feature_list = list(features.columns)
features = np.array(features)
print(features) 

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.5,random_state = 21)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

rf = RandomForestClassifier(n_estimators= 100, random_state=42)
rf_new = RandomForestClassifier(n_estimators = 100, criterion = 'mse', max_depth = None, 
                               min_samples_split = 2, min_samples_leaf = 1)

rf.fit(train_features, train_labels)
predictions = rf.predict(test_features)

errors = metrics.accuracy_score(test_labels, predictions)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels.shape[0])
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


def draw_tree(rf, feature_list=feature_list): 
    # Pull out one tree from the forest
    tree = rf.estimators_[5]

    export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
    (graph, ) = pydot.graph_from_dot_file('tree.dot')
    graph.write_png('tree.png'); 

    print('The depth of this tree is:', tree.tree_.max_depth)


draw_tree(rf)

print("###########Smaller tree")
# Limit depth of tree to 2 levels
rf_small = RandomForestClassifier(n_estimators=10,random_state=42)
rf_small.fit(train_features, train_labels)

predictions = rf_small.predict(test_features)
errors = metrics.accuracy_score(test_labels, predictions)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels.shape[0])

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

draw_tree(rf_small)


print("###########Important features")
# Get numerical feature importances
importances = list(rf.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in zip(range(5), feature_importances)];

# New random forest with only the two most important variables
rf_most_important = RandomForestClassifier(n_estimators= 10, random_state=42)

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
