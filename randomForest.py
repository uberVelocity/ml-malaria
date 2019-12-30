import json 
import os 
from sklearn.utils.multiclass import unique_labels 
import pandas as pd 
from PIL import Image 
import numpy as np 
import cv2 

entries = os.listdir(os.getcwd()+'/malaria') 
train_json = os.getcwd()+'/malaria/training.json' 
with open(train_json, 'r') as file: 
    data = file.read()

train_data = json.loads(data)

#global vairable: 
number_bin = 4 # in image description 
from sklearn.preprocessing import normalize

def imageDescriptor(im, min, max):
    out_hist = [] #historgram

    left = min['c']
    top = min['r']
    right= max['c']
    bottom =max['r']
    im_box = im.crop((left, top, right, bottom))
    height, width =im_box.size
    partial_h = int(height/number_bin) 
    partial_w = int(width/number_bin) 
    im_new = im_box
    #suft image descriptor 

    kaze = cv2.AKAZE_create() 
    kaze.setThreshold(40) 
    gray = cv2.cvtColor(np.float32(im_new), cv2.COLOR_BGR2GRAY)
    kp, des = kaze.detectAndCompute(gray, None)
    des = np.array(des) 
    des = des.ravel() 
    #print("des",des)
    counter =0 
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
    out_hist+=list(des)
    #print(out_hist)
    return out_hist


def createDataFrame(train_data): 
    out = pd.DataFrame(columns=['image','label','infection', 'histogram'])
    total_counter = 0
    not_infected =0
    counter =0 
    data_inf=0

    flag ='init'
    for index, item in zip(range(2), train_data):  
    #for item in train_data:   
        pathimage= item['image']['pathname'] 
        im = Image.open(os.getcwd()+'/malaria'+pathimage)

        for obj in item['objects']:
            infected = 'yes'
            label = obj['category'] 
            total_counter +=1

            if obj['category']=='difficult': 
                total_counter-=1 
                continue
         
            if obj['category'] == 'red blood cell': 
                infected = 'no'
                not_infected+=1
            elif obj['category'] == 'leukocyte': 
                infected = 'no'
                not_infected+=1
            
            # historgram = imageDescriptor(im, obj['bounding_box']['minimum'], obj['bounding_box']['maximum'])
            # out.loc[len(out)]= [pathimage, label, infected, historgram]

            print(obj['category'])
            
            if flag == 'addedInfected':
                historgram = imageDescriptor(im, obj['bounding_box']['minimum'], obj['bounding_box']['maximum'])
                out.loc[len(out)]= [pathimage, label, infected, historgram]
                flag = 'notyet'
                counter+=1 

            if label != 'red blood cell' or label!= 'leukocyte': 
                #create image desciptor 1) historgram 
                flag = 'addedInfected'
                historgram = imageDescriptor(im, obj['bounding_box']['minimum'], obj['bounding_box']['maximum'])
                out.loc[len(out)]= [pathimage, label, infected, historgram]
                data_inf+=1 
                counter+=1 
    
    # print("% of data sample", data_inf/counter)
    tags = out['histogram'].apply(pd.Series)
    tags = tags.rename(columns = lambda x : 'hist_' + str(x)) 
    out = pd.concat([out[:], tags[:]], axis=1)
    out = out.drop(columns=['histogram'], axis=1)
    out = out.fillna(0)
    print("% of not infected", not_infected/total_counter, 'data shape',out.shape )
    out.to_csv("malaria.csv")
    exit()
    return out


malaria_data = createDataFrame(train_data) 
print(malaria_data)
### FROM ONLINE ######### 

# Pandas is used for data manipulation
import pandas as pd

# Use numpy to convert to arrays

# Labels are the values we want to predict
labels = np.array(malaria_data['label'])

# Remove the labels from the features
# axis 1 refers to the columns
features= malaria_data.drop(columns=['image','label','infection'], axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)

# Convert to numpy array
features = np.array(features)

print(features) 

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.5,random_state = 21)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# Import the model we are using
from sklearn.ensemble import RandomForestClassifier

# Instantiate model 
rf = RandomForestClassifier(n_estimators= 100, random_state=42)

# Train the model on training data
rf.fit(train_features, train_labels);

rf_new = RandomForestClassifier(n_estimators = 100, criterion = 'mse', max_depth = None, 
                               min_samples_split = 2, min_samples_leaf = 1)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

# Calculate the absolute errors
from sklearn import metrics 

errors = metrics.accuracy_score(test_labels, predictions)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels.shape[0])

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot

# Pull out one tree from the forest
tree = rf.estimators_[5]

# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)

# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')

# Write graph to a png file
graph.write_png('tree.png'); 

print('The depth of this tree is:', tree.tree_.max_depth)

# Limit depth of tree to 2 levels
rf_small = RandomForestClassifier(n_estimators=10,random_state=42)
rf_small.fit(train_features, train_labels)

# Extract the small tree
tree_small = rf_small.estimators_[5]

# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)

(graph, ) = pydot.graph_from_dot_file('small_tree.dot')

graph.write_png('small_tree.png')

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
important_indices = [feature_list.index('hist_5'), feature_list.index('hist_3')]
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


# Import matplotlib for plotting 
import matplotlib.pyplot as plt


