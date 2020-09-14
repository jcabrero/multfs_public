import sklearn
import sklearn.datasets
import sklearn.linear_model
from sklearn.neural_network import MLPClassifier
import numpy as np
import db
from joblib import dump, load

def compute_accuracy(Y, Y_hat):
	accuracy = float((np.dot(Y.T,Y_hat) + np.dot(1-Y.T,1-Y_hat))/float(Y.size)*100)

	return accuracy

def load_training_set():
	print("Loading training set...")
	lst = db.read_csv_list("dataset_train_1.csv")[1:]
	X = np.array([i[:-1] for i in lst]).astype(np.float64)
	Y = np.array([i[-1:] for i in lst]).astype(np.float64)
	return X, Y

def load_test_set():
	print("Loading test set...")
	lst_test = db.read_csv_list("dataset_test_1.csv")[1:]
	X_test = np.array([i[:-1] for i in lst_test]).astype(np.float64)
	Y_test = np.array([i[-1:] for i in lst_test]).astype(np.float64)
	return X_test, Y_test

def fit_model(X, Y, name="default.joblib"):
	print("Fitting neural network...")
	clf = sklearn.linear_model.LogisticRegressionCV(max_iter=25000);
	clf.fit(X, np.ravel(Y));
	dump(clf, name) 
	return clf
def load_model(name="default.joblib"):
	print("Loading model...")
	clf = load(name)
	return clf

def test_model(clf, X, Y, X_test, Y_test):
	print("Testing model...")
	LR_predictions = clf.predict(X).reshape(Y.shape)
	TST_predictions = clf.predict(X_test).reshape(Y_test.shape)
	acc1 = compute_accuracy(Y, LR_predictions)
	acc2 = compute_accuracy(Y_test, TST_predictions)
	print ('Accuracy of Training Set: %0.2f%%' %(acc1) + " (percentage of correctly labelled datapoints)")
	print ('Accuracy of Test Set: %0.2f%%' %(acc2) + " (percentage of correctly labelled datapoints)")

def make_logistic_regression():
	print("Making Logistic Regression...")
	X, Y = load_training_set()
	X_test, Y_test = load_test_set()

	clf = fit_model(X, Y)
	clf =load_model()
	test_model(clf, X, Y, X_test, Y_test)

def fit_neural_network(X, Y,name="network.joblib"):
	print("Fitting neural network...")
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50, 25, 7, 3), 
		random_state=1, max_iter=1000)
	#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 5), 
		#random_state=1, max_iter=25000)
	clf.fit(X, np.ravel(Y))
	dump(clf, name) 

def test_neural_network(X, Y, X_test, Y_test, name="network.joblib"):
	print("Testing neural network...")
	clf = load_model(name)
	test_model(clf, X, Y, X_test, Y_test)

def make_neural_network():
	print ("Making neural network...")
	name = "50-25-7-3.network"
	#name = "network.joblib"
	X, Y = load_training_set()
	X_test, Y_test = load_test_set()
	#clf = fit_neural_network(X, Y, name)
	#clf =load_model("network.joblib")
	test_neural_network(X, Y, X_test, Y_test, name)
def main():
	
	make_neural_network()
	# Print accuracy


if __name__ == "__main__":
	main()
