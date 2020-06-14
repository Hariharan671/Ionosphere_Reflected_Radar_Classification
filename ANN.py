#Importing  Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset and Splitting into Input and Output
dataset =pd.read_csv('ionosphere_csv.csv')
X=dataset.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]].values
y=dataset.iloc[:,34].values

#Label Encoding the Output column
from sklearn.preprocessing import LabelEncoder
y= LabelEncoder().fit_transform(y)

#Splitting dataset into test and train set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Making Artificial Neural Network
#Importing Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
#Initializing ANN
classifier = Sequential()

#Adding input layer and first hidden layer
classifier.add(Dense(output_dim=16,init='uniform',activation='relu',input_dim=34))
classifier.add(Dropout(p=0.1))

#Adding second hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
classifier.add(Dropout(p=0.1))

#Adding output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

#Compiling ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

#Fitting ANN to training set
classifier.fit(X_train,y_train,batch_size=10,nb_epoch=10)

#Making Prediction and Evaluating model
#prediction
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# METRICS: Confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print("Confusion matrix is : ",cm)

#Accuracy
print("Accuracy of ANN is : ",(cm[0][0]+cm[1][1])*100/(cm[0][0]+cm[1][0]+cm[1][1]+cm[0][1]));
#Precision
print("Precision of ANN is : ",(cm[0][0])*100/(cm[0][0]+cm[0][1]));
#Recall
print("Recall of ANN is : ",(cm[0][0])*100/(cm[0][0]+cm[1][0]));

      

