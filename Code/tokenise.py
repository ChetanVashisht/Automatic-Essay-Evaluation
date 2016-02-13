import re
from numpy import genfromtxt
import nltk
corpus = set()

# pattern = r"(?u)\b\w\w+\b"
file = '../Inputs/Train.tsv'
POS_dict = {}

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

def POS_tagging(tokens, POS_dict):
	"""
		Parts of speech tagging done. Net count of each is returned.
	"""
	POS_dict.form_keys(POS_dict, 0)
	for pos, tag in nltk.pos_tag(tokens):
		POS_dict[tag] += 1
	return POS_dict


def spell_check(tokens, essay):
	"""

	"""

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
	for line in f1:
		# Essay cointins the essay and tokens contains the tokenised essay.
		line_break = line.strip("\n").strip("\t").split("\t")
		essay = line_break[2].strip('"')
		tokens = tokenise(essay)
		label = int(line_break[6])/2.0
		
		# [1] Bag of words
		corpus, length = Bag_of_words(tokens, corpus)
		
		# [2] Counting features
		count1 = count(tokens)

		# [3] POS Tagging
	print length    

# print file[0]
essay_collector(file, corpus)
