#there are parts of the code that are commented out, which are relevant to the project

from warnings import simplefilter
simplefilter(action = 'ignore', category = FutureWarning)

# importing libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#reading the file
data = pd.read_csv ('data/data.csv')

#summary of the data
data.info()

#first few rows
print (data.head())

#diagnosis is whether or not it is benign or malignant (so this is our target variable)
data.diagnosis.unique()

#outputs statistical details
data.describe()

data.drop ('id', axis = 1, inplace = True)
data.drop ('Unnamed: 32', axis = 1, inplace = True)

#binarizing: malignant = 1; benign = 0
data ['diagnosis'] = data ['diagnosis'].map({'M':1, 'B':0})

#standardizing
datas = pd.DataFrame (preprocessing.scale(data.iloc [:,1:32]))
datas.columns = list (data.iloc [:,1:32].columns)
datas ['diagnosis'] = data ['diagnosis']

#Looking at the number of patients with malignant and benign tumors:
datas.diagnosis.value_counts().plot(kind='bar', alpha = 0.5, facecolor = 'b', figsize=(12,6))
plt.title("Diagnosis (M=1 , B=0)", fontsize = '12')
plt.ylabel("Total Number of Patients")
plt.grid(b=True)

#seeing all the columns
print (data.columns)

#calculates mean of all data
data_mean = data[['diagnosis','radius_mean','texture_mean','perimeter_mean',
'area_mean','smoothness_mean', 'compactness_mean', 'concavity_mean',
'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']]

#heatmap to show correlation
#we find that radius_mean, perimeter_mean, area_mean, compactness_mean, concavity_mean, concave points_mean have the highest correlation with the diagnosis
plt.figure(figsize=(14,14))
foo = sns.heatmap(data_mean.corr(), vmax=1, square=True, annot=True)

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import metrics

predictors = data_mean.columns[2:11]
target = "diagnosis"

X = data_mean.loc[:,predictors]
y = np.ravel(data.loc[:,[target]])

#spliting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print ('Shape of training set : %i || Shape of test set : %i' % (X_train.shape[0],X_test.shape[0]) )

print ("Mean accuracy:")

#logistic regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

scores = cross_val_score(lr, X_train, y_train, scoring='accuracy' ,cv=10).mean()

print("Logistic regression: %s" % round(scores*100,2))


#support vector machine
from sklearn import svm

svm = svm.SVC()

scores = cross_val_score(svm, X_train, y_train, scoring='accuracy' ,cv=10).mean()

print("SVM: %s" % round(scores*100,2))


#k nearest neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

scores = cross_val_score(knn, X_train, y_train, scoring='accuracy' ,cv=10).mean()

print("KNN: %s" % round(scores*100,2))


#perceptron
from sklearn.linear_model import Perceptron

pct = Perceptron()

scores = cross_val_score(pct, X_train, y_train, scoring='accuracy' ,cv=10).mean()

print("Perceptron: %s" % round(scores*100,2))


#random forest ;)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

scores = cross_val_score(rf, X_train, y_train, scoring='accuracy' ,cv=10).mean()

print("Random Forest: %s" % round(scores*100,2))


#naive bayes
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

scores = cross_val_score(rf, X_train, y_train, scoring='accuracy' ,cv=10).mean()

print("The mean accuracy with 10 fold cross validation is %s" % round(scores*100,2))

#since LR, RF, NB, and KNN are working the best, we gon improve the parameters

#KNN - 7 is the most accurate
for i in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors = i)
    score = cross_val_score(knn, X_train, y_train, scoring='accuracy' ,cv=10).mean()
    print("N = " + str(i) + " :: Score = " + str(round(score,5)))

#random forest - 17 is most accurate
for i in range(1, 21):
    rf = RandomForestClassifier(n_estimators = i)
    score = cross_val_score(rf, X_train, y_train, scoring='accuracy' ,cv=10).mean()
    print("N = " + str(i) + " :: Score = " + str(round(score,5)))

#final - random forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=18)

rf = rf.fit(X_train, y_train)

predicted = rf.predict(X_test)

acc_test = metrics.accuracy_score(y_test, predicted)

print ('The accuracy on test data is %s' % (round(acc_test,4)))
