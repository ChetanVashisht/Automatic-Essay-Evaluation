from nltk.corpus import wordnet
from itertools import chain
import re, csv, os
from skll.metrics import kappa
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt

def joiner(word_set):
	# Joins the words
	# Join the word from the list to the old_dict and add to the new dict
	# Now rename the old_dict as new_dict
	
	#if len(word_set) == 1:
		#return word_set
	
	old_dict = word_set[-1]
	for dummy in word_set[:-1][::-1]:
		new_dict = []
		for word in dummy:
			for old in old_dict:
				new_dict.append(word + " " + old)
		old_dict = new_dict
		
	return old_dict	

def expected_content(content_model):
	"""
		The idea is to look for specific words/phrases in the sentences. This will help us realise 
		if the student has included the points in the essay. A total count of the number of such 
		repeating sentences is used as a feature. 
		
		Each point is analysed to find synonyms of different words and the words are stemmed to 
		ensure that every possibility is accounted for.
		
		Currently building the one time model.
	"""
	
	# Using the content expected we build the synonym phrase set
	phrase_set = []
	for phrase in content_model:
		words = phrase.split(' ')
		word_set = []
		
		# word_set not contains all the synonyms of all words in the phrase.
		for word in words:
			if word not in stop_words:
				dummy = []
				synonyms = wordnet.synsets(word)
				for arg in set(chain.from_iterable([syn.lemma_names() for syn in synonyms])):
					dummy.append(arg.encode('ascii', 'ignore'))
				word_set.append(dummy)
				# print dummy
			else:
				word_set.append([word])
		phrase_set.append(joiner(word_set))
	
	with open(output_file, 'w') as csvfile:
		wr = csv.writer(csvfile, delimiter = ',')
		for w in phrase_set:
			wr.writerow(w)
	
def feature_content(essay, output_file):
	"""
		This function searches the essay n-grams for matches with the expected content. 
		The more the matching, higher is the score rewarded. 
	"""
	
	# Gererate the pattern using regexp of python and the bigrams and trigrams
	pattern = r"\b\w+\b"
	tokens = re.findall(pattern, essay)
	unigrams, bigrams, trigrams, quadgrams, pentagrams = tokens, [], [], [], []
	for i in xrange(len(tokens)-1):
		bigrams.append(tokens[i] + " " + tokens[i+1])
	for i in xrange(len(tokens)-2):
		trigrams.append(tokens[i] + " " + tokens[i+1] + " " + tokens[i+2])
	for i in xrange(len(tokens)-3):
		quadgrams.append(tokens[i] + " " + tokens[i+1] + " " + tokens[i+2] + " " + tokens[i+3]) 
	for i in xrange(len(tokens)-4):
		pentagrams.append(tokens[i] + " " + tokens[i+1] + " " + tokens[i+2] + " " + tokens[i+3] + " " + tokens[i+4])
		
	# Compare the obtained ngrams with the available dictionary.
	grams = [unigrams, bigrams, trigrams]
	
	common, g_feat, prompt_feat = [], [], 0
	for gram in grams:
		count = 0
		for word in gram:
			if word in wordset:
				common.append(word)
				count += 1
			if word in prompts:
				prompt_feat += 1
		g_feat.append(count)
	
	return g_feat + [len(common)] + [prompt_feat]

def input_content(input_file):
	file = open(input_file, 'r')
	content_model = file.read().split('\n')
	expected_content(content_model)

def output_content(file, no):
	f1 = open(file, 'r')
	f1.readline()
	
	if 'test' in file:
		f2 = '../Experiment/' + str(no) + '/' + str(no) + '_test.csv'
	else:
		f2 = '../Experiment/' + str(no) + '/'+ str(no) + '_train.csv'
	
	if os.path.isfile(f2):
		os.remove(f2)
		
	for line in f1:
		line_break = line.strip("\n").strip("\t").split("\t")
		essay = line_break[2].strip('"')
		essay_id = line_break[0]
		label = int(line_break[6])
		
		row = feature_content(essay, output_file)
		
		with open(f2, 'a') as csvfile:
			# print essay_id, row, label
			writer = csv.writer(csvfile, delimiter = ',')
			writer.writerow([essay_id] + row + [label])

classifiers = [GaussianNB(), RandomForestClassifier()]
names = ['Bayes', 'Forest']
b, c = [], []
print "Essay\tType\tFeature\t\tBayes\t\tForest"
for no in [1, 3, 4, 5, 6]:
	a = [[] for x in xrange(len(classifiers))]
	output_file = '../Experiment/' + str(no) + '/content' + str(no) + '.csv'
	input_file = '../Experiment/' + str(no) + '/' + str(no) + '.txt'
	f = open('../Experiment/' + str(no) + '/prompt' + str(no) + '.txt', 'r')
	file = '../Experiment/' + str(no) + '/' + str(no) + '_train.csv'
	file1 = '../Experiment/' + str(no) +'/' + str(no) + '_test.csv'
	file3 = '../Data/Features/features_train' + str(no) + '.csv'
	file4 = '../Data/Features/features_test' + str(no) + '.csv'

	
	stop = open('stopwords.txt', 'r')
	stop_words = stop.read().split('\n')
	input_content(input_file)
	# print "Finished Extracting Content"

	"""
	Define global arrays prompts and wordset.
	"""
	prompts = f.read().split('\n')

	f1 = open(output_file, 'r')
	wordset = []
	for line in f1:
		line = line.strip('\n').strip('\r')
		line = line.split(',')
		wordset = wordset + line

	"""
	Extracting the new features.
	"""
	#output_content('../Data/Train/new_train' + str(no) + '.tsv', no)
	#print "Train Done"
	#output_content('../Data/Test/new_test' + str(no) + '.tsv', no)
	#print "Test Done"

	"""
		features_train and features_test contains the features from the content model.
		While dummy_train and dummy_test contains the original feature set.
		The combined set is in combined_train and combined_test.
		
		Original test accuracy: 81.5%, original train accuracy: 81.2%
		Content test accuracy: 13.5%, Content train accuracy: 17.2%
		Combined test accuracy: 
	"""


	dummy_train = np.genfromtxt(file3, dtype = int, delimiter = ',', skip_header = 1)
	dummy_train = np.delete(dummy_train, (1,0,14), 1)

	dummy_test = np.genfromtxt(file4, dtype = int, delimiter = ',', skip_header = 1)
	dummy_test = np.delete(dummy_test, (1,0,14), 1)

	features_train = np.genfromtxt(file, dtype = int, delimiter = ',', skip_header = 0)
	features_test = np.genfromtxt(file1, dtype = int, delimiter = ',', skip_header = 0)

	labels = features_train[:, len(features_train[1]) - 1]
	features_train = np.delete(features_train, 0, 1)
	features_train = np.delete(features_train, len(features_train[1]) - 1, 1)

	test_labels = features_test[:, len(features_test[1]) - 1]
	features_test = np.delete(features_test, 0, 1)
	features_test = np.delete(features_test, len(features_test[1]) - 1, 1)

	# print features_train.shape, dummy_train.shape
	# print features_test.shape, dummy_test.shape
	
	combined_train = np.column_stack((features_train, dummy_train))
	combined_test = np.column_stack((features_test, dummy_test))
	
	# "Essay Set 	Classifier	Feature Set		Accuracy "
	# print "Set " + str(no)
	for i, clf in enumerate(classifiers):
		clf.fit(dummy_train, labels)
		prediction = clf.predict(dummy_train)
		prediction1 = clf.predict(dummy_test)
		a[i].append(kappa(test_labels, prediction1, weights = 'quadratic'))
		a[i].append(kappa(labels, prediction, weights = 'quadratic'))
	print no,"\tTest\t Stat\t ", a[0][0],'\t',a[0][1]
	#print no,"\tTrain\t Stat\t ",a[1][0],'\t',a[1][1]

	for i, clf in enumerate(classifiers):
		clf.fit(features_train, labels)
		prediction = clf.predict(features_train)
		prediction1 = clf.predict(features_test)
		a[i].append(kappa(test_labels, prediction1, weights = 'quadratic'))
		a[i].append(kappa(labels, prediction, weights = 'quadratic'))
	print no,"\tTest\t Prompt\t ", a[0][2],'\t',a[0][3]
	#print no,"\tTrain\t Prompt\t ",a[1][2],'\t',a[1][3]

	for i, clf in enumerate(classifiers):
		clf.fit(combined_train, labels)
		prediction = clf.predict(combined_train)
		prediction1 = clf.predict(combined_test)
		a[i].append(kappa(test_labels, prediction1, weights = 'quadratic'))
		a[i].append(kappa(labels, prediction, weights = 'quadratic'))
	print no, "\tTest\t Comb\t ", a[0][4],'\t',a[0][5], '\n'
	#print no, "\tTrain\t Comb\t ",a[1][4],'\t',a[1][5], '\n'
	
	b.append(a[0])
	c.append(a[1])
	
plt.figure()
x_ticks = ['Stat', 'Prompt', 'Comb']
x = [1, 2, 3]
plt.xticks(x, x_ticks)
plt.plot(x, [a[0][0], a[0][2], a[0][4]], x, [a[0][1], a[0][3], a[0][5]])
plt.title('Essay Set 6 Training vs Testing accuracy')
plt.xlabel('Feature Type')
plt.ylabel('Accuracy')
	

plt.figure()
x_ticks = ['1', '3', '4', '5', '6']
x = [1, 2, 3, 4, 5]
plt.xticks(x, x_ticks)
plt.plot(x, [b[0][1], b[1][1], b[2][1], b[3][1], b[4][1]], x, [c[0][2], c[1][2], c[2][2], c[3][2], c[4][2]])
plt.title('Combined accuracy vs Prompt accuracy')
plt.xlabel('Essay Set number')
plt.ylabel('Accuracy')

plt.figure()
x_ticks = ['1', '3', '4', '5', '6']
x = [1, 2, 3, 4, 5]
plt.xticks(x, x_ticks)
plt.plot(x, [b[0][1], b[1][1], b[2][1], b[3][1], b[4][1]], x, [c[0][1], c[1][1], c[2][1], c[3][1], c[4][1]])
plt.title('Training vs Testing for Combined feature set')
plt.xlabel('Essay Set number')
plt.ylabel('Accuracy')
plt.show()
