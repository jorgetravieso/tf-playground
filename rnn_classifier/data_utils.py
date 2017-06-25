import re
import numpy as np


def clean_str(string):
	"""
	Tokenization/string cleaning for all datasets except for SST.
	Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
	"""
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	return string.strip().lower()


def load_data_and_labels():
	"""
	Loads MR polarity data from files, splits the data into words and generates labels.
	Returns split sentences and labels.
	"""

	data = list(open("./data/demo.txt", "r").readlines())
	labels_counter = 0
	labels = {}

	# first pass populate the labels dict
	for i, e in enumerate(data):
		tab_index = e.index('\t')
		label = e[0:tab_index]
		if label not in labels:
			labels[label] = labels_counter
			labels_counter += 1

	# second pass populate the data (x) and numeric labels (y)
	x_text = []
	y = np.zeros(shape=(len(data), labels_counter))
	for i, e in enumerate(data):
		tab_index = e.index('\t')
		label, text = e[0:tab_index], e[tab_index:]
		# print("Text: " + text.strip() + " label " + label + "label index " + str(labels[label]))
		y.itemset(i, labels[label], 1)
		x_text.append(clean_str(text))

	return [x_text, y, labels]
