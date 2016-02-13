import re
from numpy import genfromtxt
import nltk
from nltk.data import load
import enchant
import string.punctuation
corpus = set()
tagdict = load('help/tagsets/upenn_tagset.pickle')

class essay:
	def init():
		"""
			To inititalise the class. It holds all the features of the essay.
		"""
		self.tokens = tokens
		self.spell_check = spell_check
		self.c_sentence = c_sentence
		self.c_total = c_total
		self.c_long = c_long
		self.c_unique = c_unique
		self.c_punctuation = c_punctuation
		self.c_comma = c_comma
		self.c_bracket = c_bracket
		self.c_quotes = c_quotes
		self.noun = noun
		self.verb = verb
		self.adjective = adjective
		self.adverb = adverb

	def record(file):
		"""
			Documents the record into a csv file.
		"""
		f = open(file, 'w+', newline = '\n')
		write = csv.writer(f, delimiter = ',')


# TODO: 
#		Class for outputing features to csv
#		Classify the obtained features using a quadratic weighing kappa

# pattern = r"(?u)\b\w\w+\b"
file = '../Inputs/Train.tsv'

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
		[4] punctuation marks
		[5] Bracket count
	"""
	sentence = len(re.finall(r'\.', essay))
	total = len(tokens)
	unique = len(set(tokens))
	for word in tokens:
		if len(word) >= 5:
			long += 1

	comma = len(re.findall(r'\,', essay))
	bracket = len(re.findall(r'\(', essay))
	quotes = len(re.findall(r'\"'), essay)/2.0
	temp = essay.strip(string.punctuation)
	punctuation = len(essay) - temp

	return 
	


def POS_tagging(tokens, POS_dict):
	"""
		Parts of speech tagging done. Net count of each is returned.
	"""
	POS_dict.form_keys(POS_dict, 0)
	for pos, tag in nltk.pos_tag(tokens):
		POS_dict[tag] += 1
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
	f1 = open(file, 'r')
	f2 = open('../Data/features.csv', 'w')
	f1.readline()
	writer = csv.writer(f2, newline = '\n')
	writer.writerow(["Essay"] + ["spell_check"] + ["c_sentence"] + ["c_total"] + ["c_long"] 
	                + ["c_unique"] + ["c_punctuation"] + ["c_comma"] + ["c_bracket"] + ["c_quotes"]
	                + ["noun"] + ["verb"] + ["adjective"] + ["adverb"])
	f2.close()

	for line in f1:
		# Essay cointins the essay and tokens contains the tokenised essay.
		line_break = line.strip("\n").strip("\t").split("\t")
		essay = line_break[2].strip('"')
		tokens = tokenise(essay)
		label = int(line_break[6])/2.0
		POS_dict = {tagdict.keys()}
		
		# [1] Bag of words
		corpus, length = Bag_of_words(tokens, corpus)
		
		# [2] Counting features
		count1 = count(tokens)

		# [3] POS Tagging
		POS_dict = POS_tagging(tokens, POS_dict)

		# [4] Spell check
		wrong_spellings = spell_check(tokens)
	  

# print file[0]
essay_collector(file, corpus)
