#!/usr/bin/python

import sys
import pickle
import pandas as pd
from sklearn import tree
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import metrics
from pprint import pprint
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from tester import test_classifier
from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from sklearn import grid_search
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import f1_score
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

### Task 1: Select features
features_list = ['poi','salary','fraction_to_poi','fraction_from_poi',
                 'bonus','exercised_stock_options', 'long_term_incentive', 
                 'restricted_stock','shared_receipt_with_poi', 'total_payments',
                 'total_stock_value','deferral_payments','deferred_income',
                 'expenses','other','restricted_stock_deferred','director_fees',
                 'loan_advances']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
### Task 2: Remove 'obvious' outliers
data_dict.pop('TOTAL',0)  
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Task 3: Create new feature(s)
#I am creating two new features that turn the to_poi and from_poi email features and ratio

for person in my_dataset:
    if my_dataset[person]['from_poi_to_this_person'] == "NaN" \
    or my_dataset[person]['to_messages'] == "NaN":
        my_dataset[person]['fraction_from_poi'] = "NaN"
    else:
        my_dataset[person]['fraction_from_poi'] \
         = float(my_dataset[person]['from_poi_to_this_person']) \
         / my_dataset[person]['to_messages'] 

for person in my_dataset:
    if my_dataset[person]['from_this_person_to_poi'] == "NaN" \
    or my_dataset[person]['from_messages'] == "NaN":
        my_dataset[person]['fraction_to_poi'] = "NaN"
    else:
        my_dataset[person]['fraction_to_poi'] \
        = float(my_dataset[person]['from_this_person_to_poi']) \
        / my_dataset[person]['from_messages']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#split the data for training and testing purposes
features_train, features_test, labels_train, labels_test \
= train_test_split(features, labels, test_size=0.3, random_state=42)


#create a cross validator  using StratifiedShuffleSplit
cv_sss = StratifiedShuffleSplit(labels, 100, random_state=42)

### Task 4: Create a classifiers using pipeline, MixMaxScaller, 
### KBest(with Chi squared) and Decision Tree
pipeline_DT = Pipeline([('MinMax', MinMaxScaler()),('kbest', SelectKBest(chi2)),
('clf', tree.DecisionTreeClassifier(random_state = 42)),])
pipeline_DT = pipeline_DT.fit(features_train, labels_train)
pred_DT = pipeline_DT.predict(features_test)

#Preliminary check of Decision Tree classifier scores with test data and a review 
###of parameters selected
print "DT Accuracy Score"
print pipeline_DT.score(features_test, labels_test)
print "DT Precision score: "
print precision_score(labels_test, pred_DT, average = 'weighted')
print "DT Recall score "
print recall_score(labels_test, pred_DT, average = 'weighted')

print "\nInitial DT Pipeline parameters:"
pprint(pipeline_DT.get_params())

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
'''
Use GridSearCV to optimize parameters for classifier. Use F1 scoring to figure out best 
parameters and use my StratifiedShuffleSplit data to cross validate
'''
param_grid_DT = {'clf__min_samples_split': [2, 3, 4, 5],
                 'clf__max_depth': [4, 5, 6, 7, 8, None],
                 'clf__random_state': [42, 53],
                 "kbest__k" : [2, 4, 6, 8, 10, 12]}

gridCV = GridSearchCV(pipeline_DT, param_grid = param_grid_DT, scoring="f1", cv = cv_sss)

gridCV.fit(features, labels)

#review best parameters used in tuned classifier
print "Best parameters from parameter grid:"
print gridCV.best_params_

#get the features used and scores
X_new = gridCV.best_estimator_.named_steps['kbest']

# Get SelectKBest scores, rounded to 2 decimal places, name them "feature_scores"
feature_scores = ['%.2f' % elem for elem in X_new.scores_ ]
# Get SelectKBest pvalues, rounded to 3 decimal places, name them "feature_scores_pvalues"
feature_scores_pvalues = ['%.3f' % elem for elem in  X_new.pvalues_ ]
# Get SelectKBest feature names, whose indices are stored in 'X_new.get_support',
# create a tuple of feature names, scores and pvalues, name it "features_selected_tuple"
features_selected_tuple=[(features_list[i+1], feature_scores[i], 
                          feature_scores_pvalues[i]) \
                         for i in X_new.get_support(indices=True)]
# Sort the tuple by score, in reverse order
features_selected_tuple = sorted(features_selected_tuple,
                                 key=lambda feature: float(feature[1]) , reverse=True)
# Print
print ' '
print 'Selected Features, Scores, P-Values'
print features_selected_tuple

#create the final classifer to use in the test
clf = gridCV.best_estimator_

#test classifier
test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list

dump_classifier_and_data(clf, my_dataset, features_list)