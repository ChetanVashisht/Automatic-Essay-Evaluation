from numpy import vstack, delete, genfromtxt, hstack, asarray, savetxt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from skll.metrics import kappa

train_file = open('../Data/Train/new_train1.tsv', 'r')
test_file = open('../Data/Test/new_test1.tsv', 'r')

feature_train = genfromtxt('../Data/Features/features_train1.csv', dtype = int, delimiter = ',', skip_header = 1)
feature_test = genfromtxt('../Data/Features/features_test1.csv', dtype = int, delimiter = ',', skip_header = 1)

write_train = '../Experiment/features_train.csv'
write_test = '../Experiment/features_test.csv'

advantages = ['impact', 'field', 'quick', 'access', 'easy', 'automate', 'time', 'effort', 'money', 'effective', 'efficient', 'records', 'transactions', 'books', 'movies', 'games', 'songs', 'time pass', 'social media', 'communication', 'information', 'sharing', 'education', 'language', 'programming', 'internet', 'websites', 'prediction', 'processing']
disadvantages = ['unemployment', 'lack of purpose', 'waste of time', 'waste of energy', 'bad for health', 'strain', 'data security', 'computer crimes', 'piracy', 'privacy', 'heacker', 'health risk', 'hazard', 'e-waste', 'social media']

good = ['benifit', 'moderation', 'agree']
bad = ['not', 'disagree']

dummy1, dummy2 = [], []
train_file.readline()
test_file.readline()		# To skip the header file.

files = [train_file, test_file]
feature = [feature_train, feature_test]
write_files = [write_train, write_test]

addition = open('words/addition.txt', 'r').readline().split(',')
consequence = open('words/consequence.txt', 'r').readline().split(',')
contrast = open('words/contrast.txt', 'r').readline().split(',')
direction = open('words/direction.txt', 'r').readline().split(',')
diversion = open('words/diversion.txt', 'r').readline().split(',')
emphasis = open('words/emphasis.txt', 'r').readline().split(',')
exception = open('words/exception.txt', 'r').readline().split(',')
exemplifying = open('words/exemplifying.txt', 'r').readline().split(',')
generalizing = open('words/generalizing.txt', 'r').readline().split(',')
illustration = open('words/illustration.txt', 'r').readline().split(',')
similarity = open('words/similarity.txt', 'r').readline().split(',')
restatement = open('words/restatement.txt', 'r').readline().split(',')
sequence = open('words/sequence.txt', 'r').readline().split(',')
summarizing = open('words/summarizing.txt', 'r').readline().split(',')

t_words = [addition, consequence, contrast, direction, diversion, emphasis, exception, exemplifying, generalizing, illustration, similarity, restatement, sequence, summarizing]

for file, feat, write in zip(files, feature, write_files):
	dummy = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	for line in file:
		essay = line.strip('\n').strip('\t').split('\t')[2]
		sentences = essay.split('.')

		introduction_bad = []
		introduction_good = []
		body_good = []
		body_bad = []
		conclusion_good = []
		conclusion_bad = []

		for word in sentences[0].split(' '):
			if word in good:
				introduction_good.append(word)
			elif word in bad:
				introduction_bad.append(word)

		for sentence in sentences[1:-1]:
			for word in sentence.split(' '):
				if word in advantages:
					body_good.append(word)
				elif word in disadvantages:
					body_bad.append(word)

		for word in sentences[-1].split(' '):
			if word in good:
				conclusion_good.append(word)
			elif word in bad:
				conclusion_bad.append(word)

		wrapper = [len(set(introduction_good))]
		wrapper.append(len(set(introduction_bad)))
		wrapper.append(len(set(body_good)))
		wrapper.append(len(set(body_bad)))
		wrapper.append(len(set(conclusion_good)))
		wrapper.append(len(set(conclusion_bad)))

		unigrams = essay.strip('.').split(' ')
		bigrams = [" ".join((unigrams[i], unigrams[i+1])) for i in xrange(len(unigrams) - 2)]
		trigrams = [" ".join((unigrams[i], unigrams[i+1], unigrams[i+2])) for i in xrange(len(unigrams) - 3)]
		grams = [unigrams, bigrams, trigrams]
		
		for transition in t_words:
			count = 0
			for gram in grams:
				for word in gram:
					if word in transition:
						count += 1
			wrapper.append(count)


		dummy = vstack((dummy, wrapper))
		# print line.strip('\n').strip('\t').split('\t')[0]

	dummy = delete(dummy, 0, 0)
	dummy = asarray(dummy)



	# if file == train_file:
	# 	dummy1 = dummy
	# else:
	# 	dummy2 = dummy

	print dummy.shape, feat.shape
	dummy = hstack((feat, dummy))
	savetxt(write, dummy, delimiter = ',')

#------------------------------------------------------------------------------------------#
train = genfromtxt('../Experiment/features_train.csv', delimiter = ',', skip_header = 1)
train_labels = train[:, 15]
train = delete(train, 15, 1)

test = genfromtxt('../Experiment/features_test.csv', delimiter = ',', skip_header = 1)
test_labels = test[:, 15]
test = delete(test, 15, 1)

rf = RandomForestClassifier(n_estimators = 100, n_jobs = 2)
rf.fit(train, train_labels)
results_boosting = rf.predict(test)

clf = GaussianNB()
clf.fit(train, train_labels)
prediction = clf.predict(test)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train, train_labels)
results_KNN = neigh.predict(test)

print 'Forest', kappa(test_labels, results_boosting, weights = 'quadratic')
print 'Bayes', kappa(test_labels, prediction, weights = 'quadratic')
print 'KNN', kappa(test_labels, results_KNN, weights = 'quadratic')

#-------------------------------------------------------------------------------------------#
"""
train = genfromtxt('../Data/Features/features_train1.csv', dtype = int, delimiter = ',', skip_header = 1)
train_labels = train[:, -1]
train = delete(train, -1, 1)

test = genfromtxt('../Data/Features/features_test1.csv', dtype = int, delimiter = ',', skip_header = 1)
test_labels = test[:, -1]
test = delete(test, -1, 1)

clf = GaussianNB()
clf.fit(train, train_labels)
prediction = clf.predict(test)
print kappa(test_labels, prediction, weights = 'quadratic')
"""
#-------------------------------------------------------------------------------------------#
"""
clf = GaussianNB()
clf.fit(dummy1, train_labels)
prediction = clf.predict(dummy2)
print kappa(test_labels, prediction, weights = 'quadratic')
"""