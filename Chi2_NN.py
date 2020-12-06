import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support, recall_score, precision_score, f1_score, confusion_matrix, accuracy_score
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

import seaborn as sn
import matplotlib.pyplot as plt

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

np.random.seed(42)

## Load data and labels ###

label = pd.read_csv('dataset/labels.csv')
data = pd.read_csv('dataset/data.csv')

#Encode the variable : 

X = data.values[:,1:]
X = np.asarray(X).astype('float32')

#Encode the variable 

encode = preprocessing.LabelEncoder()
encode.fit(label.Class.unique())
y = encode.transform(label.Class.values)

y = np.array(y)
#encode y to categorical
encoded = to_categorical(y)
y = encoded


#NN model
def create_network(n_feats):
    model = Sequential()
    model.add(Dense(128, input_dim = n_feats, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    # compile the keras model
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    return model

KF = KFold(n_splits=5,shuffle=True)
itern =0
total_train_accuracy = 0
total_test_accuracy = 0
total_precision = 0
total_recall = 0
total_fscore = 0
final_conf_mat =0

for train_index, test_index in KF.split(X):
	# Split train-test
	x_train, x_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]

	#Feature Selection with chi2 test
	ch2 = SelectKBest(chi2, k=20)
	x1_train = ch2.fit_transform(x_train, y_train)
	x1_test = ch2.transform(x_test)

	### Classification model (ADD NN to be fit here)###

	classifier = create_network(x1_train.shape[1])
	classifier.fit(x1_train, y_train, epochs=100, batch_size=25, validation_data=(x1_test, y_test))

	y_pred = classifier.predict(x1_test)

	#### Scoring #####
	print ("\nAccuracy on Training Set :")
	_, accuracy_train = classifier.evaluate(x1_train, y_train)
	print('Accuracy: %.2f' % (accuracy_train*100))
	#print("____________")
	#print (accuracy_score(inverted_y_test,inverted_y_pred))
	total_train_accuracy += accuracy_train

	print ("\nChecking on Test Set")
	print ("\nAccuracy on Testing Set :")
	_, accuracy_test = classifier.evaluate(x1_test, y_test)
	print('Accuracy: %.2f' % (accuracy_test*100))
	total_test_accuracy += accuracy_test


	##### Decoding the labels ####

	#inverting to_categorical encoding
	inverted_y_pred = (np.argmax(y_pred, axis=1)).reshape(-1,1)
	inverted_y_test = (np.argmax(y_test, axis=1)).reshape(-1,1)

	###Scoring ###

	print("Fold no: "+str(itern))
	itern+=1
	
	print ("Checking on Test Set")
	print ("\nAccuracy on Testing Set :"+str(accuracy_score(inverted_y_test,inverted_y_pred)))
	#total_test_accuracy += accuracy_score(y_test,y_pred)
  
	total_precision+= precision_score(inverted_y_test, inverted_y_pred,average='macro')
	total_recall+= recall_score(inverted_y_test, inverted_y_pred,average='macro')	
	total_fscore+= f1_score(inverted_y_test, inverted_y_pred,average='macro')	
	print ("\nPrecision Score")
	print (precision_score(inverted_y_test, inverted_y_pred,average='macro'))
	print ("\nRecall Score")
	print (recall_score(inverted_y_test, inverted_y_pred,average='macro'))	
	print ("\nF1 Score")
	print (f1_score(inverted_y_test, inverted_y_pred,average='macro'))

	#Confusion Matrix
	conf_mat=confusion_matrix(inverted_y_test, inverted_y_pred)
	final_conf_mat+=conf_mat
	print("Confusion matrix :\n")
	print(conf_mat)
    
print("Mean train accuracy : %.2f" % ((total_train_accuracy/5)*100))
print("Mean test accuracy : %.2f" % ((total_test_accuracy/5)*100))
print("Mean precision : %.2f" % ((total_precision/5)*100))
print("Mean recall : %.2f" % ((total_recall/5)*100))
print("Mean fscore : %.2f" % ((total_fscore/5)*100))

print("Confusion matrix : " )
print(final_conf_mat)
#plt.figure(figsize=(5,5))
sn.heatmap(final_conf_mat, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()




