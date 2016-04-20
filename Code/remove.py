from random import random
from skll.metrics import kappa

def remover(file):
	f = open(file)
	f.readline()	# To skip header
	
	for row in f:
		row1 = row.strip().split("\t")
		essay_set = row1[1]

		writer_train = open('../Inputs/Train/new_train' + str(essay_set) + '.tsv', 'a')
		writer_test = open('../Inputs/Test/new_test' + str(essay_set) + '.tsv', 'a')
		
		if essay_set != '2':
			if (random() >= 0.65):
				writer_test.write(row)
			else:
				writer_train.write(row)

		writer_train.close()
		writer_test.close()
		print essay_set
	
	f.close()


def testing(file):
	""" 
		To test and see if the quadratic weighing kappa function is working properly
	"""
	f = open(file, 'r')
	f.readline()

	labels, estimate = [], []
	for row in f:
		label = row.strip().split("\t")[6]
		if random() > 0.5:
			estimate.append(int(4*int(label)*random()))
		else:
			estimate.append(int(int(label)*random()))
		labels.append(int(label))

	print kappa(labels, labels, weights = 'quadratic')

remover('../Inputs/Train.tsv')

# testing('../Inputs/new_test.tsv')