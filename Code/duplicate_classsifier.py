import numpy as np
from sklearn.svm import SVC
from skll.metrics import kappa
from sklearn.naive_bayes import GaussianNB

file = '../Data/features.csv'
file1 = '../Data/features_test.csv'
features = np.genfromtxt(file, dtype = int, delimiter = ',', skip_header = 1)
features1 = np.genfromtxt(file1, dtype = int, delimiter = ',', skip_header = 1)

essay_sets = [1, 3, 4, 5, 6, 7, 8]
for essay_set in essay_sets:
	features_train = features[features[:, 1] == essay_set, :]
	labels = features_train[:, len(features[1]) - 1]
	features_train = np.delete(features_train, len(features_train[1]) - 1, 1)
	clf = GaussianNB()
	clf.fit(features_train, labels)

	features_test = features1[features1[:, 1] == essay_set, :]
	test_labels = features_test[:, len(features[1]) - 1]
	features_test = np.delete(features_test, len(features_test[1]) - 1, 1)

	prediction = clf.predict(features_test)
	
	print essay_set, kappa(test_labels, prediction, weights = 'quadratic')


"""
f = open('new_train.tsv', 'r')
a = f.readline().strip('\n').strip('\t').split('\t')
advantages = ['impact', 'field', 'quick', 'access', 'easy', 'automate', 'time', 'effort', 'money', 'effective', 'efficient', 'records', 'transactions', 'books', 'movies', 'games', 'songs', 'time pass', 'social media', 'communication', 'information', 'sharing', 'education', 'language', 'programming', 'internet', 'websites', 'prediction', 'processing']
disadvantages = ['unemployment', 'lack of purpose', 'waste of time', 'waste of energy', 'bad for health', 'strain', 'data security', 'computer crimes', 'piracy', 'privacy', 'heacker', 'health risk', 'hazard', 'e-waste', 'social media']
for word in sentences[0]:
	if word in introduction_bad:
		print True
"""