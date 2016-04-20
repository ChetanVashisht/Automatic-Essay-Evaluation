import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from skll.metrics import kappa

print '\tForest\t\tBayes\t\tKNN'
essay_sets = [1, 3, 4, 5, 6, 7, 8]
for i in essay_sets:
	file = '../Data/Features/features_train' + str(i) + '.csv'
	file1 = '../Data/Features/features_test' + str(i) + '.csv'
	features = np.genfromtxt(file, dtype = int, delimiter = ',', skip_header = 1)
	features1 = np.genfromtxt(file1, dtype = int, delimiter = ',', skip_header = 1)

	train_labels = features[:, len(features[1]) - 1]
	train = np.delete(features, len(features[1]) - 1, 1)

	test_labels = features1[:, len(features1[1]) - 1]
	test = np.delete(features1, len(features1[1]) - 1, 1)

	rf = RandomForestClassifier(n_estimators = 100, n_jobs = 2)
	rf.fit(train, train_labels)
	results_boosting = rf.predict(test)

	clf = GaussianNB()
	clf.fit(train, train_labels)
	prediction = clf.predict(test)

	neigh = KNeighborsClassifier(n_neighbors=3)
	neigh.fit(train, train_labels)
	results_KNN = neigh.predict(test)

	print i, '\t', kappa(test_labels, results_boosting, weights = 'quadratic'), '\t', kappa(test_labels, prediction, weights = 'quadratic'), '\t', kappa(test_labels, results_KNN, weights = 'quadratic')
