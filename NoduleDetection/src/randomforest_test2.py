'''
Created on 16-mrt.-2014

@author: Eigenaar
'''
# from sklearn.ensemble import RandomForestClassifier
# X = [[0, 0], [1, 1]]
# Y = [0, 1]
# clf = RandomForestClassifier(n_estimators=10)
# clf = clf.fit(X, Y)
# print(clf.)


# from sklearn.ensemble import RandomForestClassifier
#  import csv_io
#  import scipy
# 
# def main():
#  #read in the training file
#  train = csv_io.read_data("train.csv")
#  #set the training responses
#  target = [x[0] for x in train]
#  #set the training features
#  train = [x[1:] for x in train]
#  #read in the test file
#  realtest = csv_io.read_data("test.csv")
# 
# # random forest code
#  rf = RandomForestClassifier(n_estimators=150, min_samples_split=2, n_jobs=-1)
#  # fit the training data
#  print('fitting the model')
#  rf.fit(train, target)
#  # run model against test data
#  predicted_probs = rf.predict_proba(realtest)
# 
# predicted_probs = ["%f" % x[1] for x in predicted_probs]
#  csv_io.write_delimited_file("random_forest_solution.csv", predicted_probs)
# 
# print ('Random Forest Complete! You Rock! Submit random_forest_solution.csv to Kaggle')
# 
# if __name__=="__main__":
#  main()

from sklearn.cross_validation import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

X, y = make_blobs(n_samples=10000, n_features=10, centers=100, random_state=0)
clf = DecisionTreeClassifier(max_depth=None, min_samples_split=1, random_state=0)
scores = cross_val_score(clf, X, y)
scores.mean()                             

clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
scores = cross_val_score(clf, X, y)
scores.mean()                             
clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
scores = cross_val_score(clf, X, y)
print(scores.mean())
