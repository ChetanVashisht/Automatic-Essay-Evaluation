import re, csv
from numpy import genfromtxt
import nltk
from nltk.data import load
import enchant
from string import punctuation
corpus = set()
tagdict = load('help/tagsets/upenn_tagset.pickle')

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

class record:
	def __init__(self, test, essay_id, essay_set, tokens, spell_check, c_sentence, c_total, c_long, c_unique, c_comma,
	         c_bracket, c_quotes, noun, verb, adjective, adverb, label, addition, consequence, contrast, direction, diversion,
	         emphasis, exception, exemplifying, generalizing, illustration, similarity, restatement, sequence, summarizing):
		"""
			To inititalise the class. It holds all the features of the essay.
		"""
		self.test = test
		self.essay_id = essay_id
		self.essay_set = essay_set
		self.tokens = tokens
		self.spell_check = spell_check
		self.c_sentence = c_sentence
		self.c_total = c_total
		self.c_long = c_long
		self.c_unique = c_unique
		self.c_comma = c_comma
		self.c_bracket = c_bracket
		self.c_quotes = c_quotes
		self.noun = noun
		self.verb = verb
		self.adjective = adjective
		self.adverb = adverb
		self.label = label
		self.addition = addition
		self.consequence = consequence
		self.contrast = contrast
		self.direction = direction
		self.diversion = diversion
		self.emphasis = emphasis
		self.exception = exception
		self.exemplifying = exemplifying
		self.generalizing = generalizing
		self.illustration = illustration
		self.similarity = similarity
		self.restatement = restatement
		self.sequence = sequence
		self.summarizing = summarizing

	def to_csv(self):
		"""
			Documents the record into a csv file.
		"""
		file = '../Data/Features/features_' + self.test + str(self.essay_set) + '.csv'
		row = [self.essay_id]
		row.append(self.essay_set)
		row.append(self.spell_check)
		row.append(self.c_sentence)
		row.append(self.c_total)
		row.append(self.c_long)
		row.append(self.c_unique)
		row.append(self.c_comma)
		row.append(self.c_bracket)
		row.append(self.c_quotes)
		row.append(self.noun)
		row.append(self.verb)
		row.append(self.adjective)
		row.append(self.adverb)
		row.append(self.addition)
		row.append(self.consequence)
		row.append(self.contrast)
		row.append(self.direction)
		row.append(self.diversion)
		row.append(self.emphasis)
		row.append(self.exception)
		row.append(self.exemplifying)
		row.append(self.generalizing)
		row.append(self.illustration)
		row.append(self.similarity)
		row.append(self.restatement)
		row.append(self.sequence)
		row.append(self.summarizing)
		row.append(self.label)

		with open(file, 'a') as csvfile:
			writer = csv.writer(csvfile, delimiter = ',')
			writer.writerow(row)

def tokenise(essay, pattern = r"\b\w+\b"):
	""" Tokenise the input essay"""
	return re.findall(pattern, essay)

def Bag_of_words(tokens, corpus):
	""" 
		Build a bag of words model updating the corpus with each essay
	"""
	token_list = set(tokens)
	corpus = corpus.union(token_list)
	return corpus

def count(tokens, essay):
	""" 
		Basic Counting fetures etracted here
		[1] Total word count
		[2] unique word count
		[3] Long word count
		[4] Bracket count
	"""
	sentence = len(re.findall(r'\.', essay))
	total = len(tokens)
	unique = len(set(tokens))
	long_word = 0
	for word in tokens:
		if len(word) >= 5:
			long_word += 1

	comma = len(re.findall(r'\,', essay))
	bracket = len(re.findall(r'\(', essay))
	quotes = len(re.findall(r'\"', essay))/2.0
	# temp = len(essay.strip(punctuation))
	# punctuation_count = len(essay) - temp

	return sentence, total, long_word, unique, comma, bracket, quotes
	
def transition_phrases(essay):
	unigrams = essay.strip('.').split(' ')
	bigrams = [" ".join((unigrams[i], unigrams[i+1])) for i in xrange(len(unigrams) - 2)]
	trigrams = [" ".join((unigrams[i], unigrams[i+1], unigrams[i+2])) for i in xrange(len(unigrams) - 3)]
	grams = [unigrams, bigrams, trigrams]
	
	wrapper = []
	for transition in t_words:
		count = 0
		for gram in grams:
			for word in gram:
				if word in transition:
					count += 1
		wrapper.append(count)

	return wrapper

def POS_tagging(essay):
	"""
		Parts of speech tagging done. Net count of each is returned.
	"""
	POS_dict = {}
	for i, j in nltk.pos_tag(essay):
		if j in POS_dict:
			POS_dict[j] += 1
		else:
			POS_dict[j] = 1

	tagdict = load('help/tagsets/upenn_tagset.pickle')
	for i in tagdict:
		if i not in POS_dict:
			POS_dict[i] = 0

	return POS_dict


def spell_check(tokens):
	"""
		Spell checker using PyEnchant
	"""
	count = 0
	d = enchant.Dict("en_US")
	for i in tokens:
		if d.check(i) == False:
			count += 1
	return count

def essay_collector(file, corpus):
	""" 
		Processes each essay for
			[1] Bag of words
			[2] Various counts (numerical features)
			[3] POS Tagging
			[4] Spell check
			[5] Punctuation count

	"""
	counter = 0		
	f1 = open(file, 'r')
	f1.readline()

	for line in f1:
		# Essay cointins the essay and tokens contains the tokenised essay.
		line_break = line.strip("\n").strip("\t").split("\t")
		essay = line_break[2].strip('"')
		essay_id = line_break[0]
		essay_set = line_break[1]
		tokens = tokenise(essay)
		label = int(line_break[6])/2
		POS_dict = tagdict
		if 'test' in file:
			test = 'test'
		else:
			test = 'train'

		# [1] Bag of words
		# corpus, length = Bag_of_words(tokens, corpus)
		
		# [2] Counting features
		sentence, total, long_word, unique, comma, bracket, quotes = count(tokens, essay)

		# [3] POS Tagging
		POS_dict = POS_tagging(essay)
		noun = POS_dict['NN'] + POS_dict['NNP'] + POS_dict['NNPS'] + POS_dict['NNS']
		verb = POS_dict['VB'] + POS_dict['VBD'] + POS_dict['VBG'] + POS_dict['VBN'] + POS_dict['VBP'] + POS_dict['VBZ']
		adjective = POS_dict['JJ'] + POS_dict['JJR'] + POS_dict['JJS']
		adverb = POS_dict['RB'] + POS_dict['RBS'] + POS_dict['RP']
		# [4] Spell check
		wrong_spellings = spell_check(tokens)

		wrapper = transition_phrases(essay)

		# Store the features into a csv file
		dummy = record(test, essay_id, essay_set, tokens, wrong_spellings, sentence, total, long_word, unique, 
		               comma, bracket, quotes, noun, verb, adjective, adverb, label, wrapper[0], wrapper[1], wrapper[2],
		               wrapper[3], wrapper[4], wrapper[5], wrapper[6], wrapper[7], wrapper[8], wrapper[9], wrapper[10],
		               wrapper[11], wrapper[12], wrapper[13])
		dummy.to_csv()
		counter += 1
		print test, counter, essay_set

# print file[0]
essay_file = [1, 3, 4, 5, 6, 7, 8]
file_type = ['train', 'test']
for j in file_type:
	for i in essay_file:
		f2 = '../Data/Features/features_' + j + str(i) + '.csv'
		with open(f2, 'w') as csvfile:
			writer = csv.writer(csvfile, delimiter = ',')
			writer.writerow(["essay_id"] + ["essay_set"] + ["spell_check"] + ["c_sentence"] + ["c_total"] + ["c_long"] 
		                + ["c_unique"] + ["c_comma"] + ["c_bracket"] + ["c_quotes"]
		                + ["noun"] + ["verb"] + ["adjective"] + ["adverb"] + ["score"] + ['Addition'] +
		                  ['Consequence'] + ['Contrast'] + ['Direction'] + ['Diversion'] + ['Emphasis'] +
		                  ['Exception'] + ['Exemplifying'] + ['Generalizing'] + ['Illustration'] + ['Similarity'] +
		                  ['Restatement'] + ['Sequence'] + ['Summarizing'])


for i in essay_file:
	file = '../Data/Train/new_train' + str(i) + '.tsv'
	essay_collector(file, corpus)
	file = '../Data/Test/new_test'+ str(i) + '.tsv'
	essay_collector(file, corpus)
 