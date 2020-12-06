import numpy as np
import pandas as pd
import pyswarms as ps
from sklearn.metrics import mean_squared_error
from sklearn import neighbors
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support, recall_score, precision_score, f1_score, confusion_matrix, accuracy_score
from sklearn import preprocessing
import seaborn as sn
import matplotlib.pyplot as plt

np.random.seed(42)

## Load data and labels ###

label = pd.read_csv('dataset/labels.csv')
data = pd.read_csv('dataset/data.csv')

#y =  label.Class.values
X = data.values[:,1:]

#Encode the variable : 

encode = preprocessing.LabelEncoder()
encode.fit(label.Class.unique())
y = encode.transform(label.Class.values)


### PSO function ###

def pso_feature_selection(X,y):

	### Change this to NN here ####
	classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
	total_feat = X.shape[1]
	
	# Define objective function
	def f_per_particle(m):

		#total_features = total_feat
		# Get the subset of the features from the binary mask
		if np.count_nonzero(m) == 0:
			X_subset = X
		else:
			X_subset = X[:,m==1]
		# Perform classification and store performance in P
		classifier.fit(X_subset, y)

		# Compute for the objective function
		P = (classifier.score(X_subset,y))
		#j = (alpha * (1.0 - P)+ (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))

		return P


	def f(x):
		n_particles = x.shape[0]
		j = [f_per_particle(x[i]) for i in range(n_particles)]
		return np.array(j)


	options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 30, 'p':2}

	# Call instance of PSO
	dimensions = total_feat # dimensions should be the number of features
	#init = np.random.choice([0, 1], size=(10,dimensions), p=[(dimensions-50)/dimensions, 50/dimensions])
	optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=dimensions, options=options,init_pos=None)

	# Perform optimization
	cost, pos = optimizer.optimize(f,iters=10,verbose=True)

	return pos


## Cross validation ###

KF = KFold(n_splits=5,shuffle=True)
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

	#### Feature Selection

	pos = pso_feature_selection(x_test,y_test)
	print(np.count_nonzero(pos))

	x1_train = np.array(x_train[:,pos==1]) 
	x1_test = np.array(x_test[:,pos==1]) 

	### Classification model (ADD NN to be fit here)###

	classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
	classifier.fit(x1_train, y_train)

	y_pred = classifier.predict(x1_test)

	##### Scoring #########

	print ("Checking on Train Set")
	print ("\nAccuracy on Training Set :"+str(classifier.score(x1_train,y_train)))
	total_train_accuracy += classifier.score(x1_train,y_train)

	print ("Checking on Test Set")
	print ("\nAccuracy on Testing Set :"+str(accuracy_score(y_test,y_pred)))
	total_test_accuracy += accuracy_score(y_test,y_pred)
  
	total_precision+= precision_score(y_test, y_pred,average='macro')
	total_recall+= recall_score(y_test, y_pred,average='macro')	
	total_fscore+= f1_score(y_test, y_pred,average='macro')	
	print ("\nPrecision Score")
	print (precision_score(y_test, y_pred,average='macro'))
	print ("\nRecall Score")
	print (recall_score(y_test, y_pred,average='macro'))	
	print ("\nF1 Score")
	print (f1_score(y_test, y_pred,average='macro'))

	#Confusion Matrix
	conf_mat=confusion_matrix(y_test, y_pred)
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