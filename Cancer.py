

import pandas as pd

#importing the dataset
# Made a seperate excel file from the provided excel sheet before importing


filepath= r'C:\Users\DELL\Desktop\Data Science - Intern\cancer.xlsx'
data= pd.read_excel(filepath)
Dataset= pd.DataFrame(data)


# data preprocessing
# Converting 4 and 2 to 1 and 0 for proper classification purposes

Dataset['Class: (2 for benign,  4 for malignant)'] = Dataset['Class: (2 for benign,  4 for malignant)'].map({4: 1, 2: 0})
Dataset = Dataset[~Dataset['Bare Nuclei'].isin(['?'])]
X = Dataset.iloc[:,1:10]
Y= Dataset.iloc[:,10]

#Dividing the dataset into training and testing

from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

#Scaling the dataset

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# Importing libraries for creating a neural network
import keras
from keras.models import Sequential
from keras.layers import Dense

# BUilding up the neural network

classifier = Sequential()
classifier.add(Dense(output_dim=5, init='uniform', activation= 'relu', input_dim=9))
classifier.add(Dense(output_dim=5, init='uniform', activation= 'relu'))
classifier.add(Dense(output_dim=1, init='uniform', activation= 'sigmoid'))

# Compiling and fitting the ANN to the given data
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X_train,Y_train,batch_size=20,nb_epoch=200)

# Predicting and setting a threshold (0.5 here)
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred>0.05)

# Various matrices for showcasing accuracy
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(Y_test,Y_pred))  
print(classification_report(Y_test,Y_pred)) 
print(accuracy_score(Y_test,Y_pred))































