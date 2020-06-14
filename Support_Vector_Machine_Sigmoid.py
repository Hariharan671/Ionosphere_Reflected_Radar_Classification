#Importing  Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset and Splitting into Input and Output
dataset =pd.read_csv('ionosphere_csv.csv')
X=dataset.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]].values
y=dataset.iloc[:,34].values

# Splitting dataset into test set and train set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2,random_state=0)

#Standard Scaling of Input columns
from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#Defining the classification Model
from sklearn.svm import SVC
classifier = SVC(kernel ='sigmoid', random_state=0)
classifier.fit(X_train,y_train)

#Making Predictions
Y_pred = classifier.predict(X_test)

# METRICS: Confusion matrix
from  sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,Y_pred)
print("Confusion matrix is : ",cm)

#Accuracy
print("Accuracy of the Model is : ",(cm[0][0]+cm[1][1])*100/(cm[0][0]+cm[1][0]+cm[1][1]+cm[0][1]));
#Precision
print("Precision of Model is : ",(cm[0][0])*100/(cm[0][0]+cm[0][1]));
#Recall
print("Recall of Model is : ",(cm[0][0])*100/(cm[0][0]+cm[1][0]));


