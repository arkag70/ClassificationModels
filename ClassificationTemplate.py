#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
data = pd.read_csv('.csv')
X = 
y = 

#split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#perform feature scaling if necessary
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#create your classifier
from sklearn. import 
classifier = 
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

#measure accuracy
from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix(y_test,y_pred)
