import numpy as np
import re
import tensorflow as tf
import itertools
from collections import Counter


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
	# # Load data from files
	# positive_examples = list(open("./data/rt-polaritydata/rt-polarity.pos", "r").readlines())
	# positive_examples = [s.strip() for s in positive_examples]
	# negative_examples = list(open("./data/rt-polaritydata/rt-polarity.neg", "r").readlines())
	# negative_examples = [s.strip() for s in negative_examples]
	# # Split by words
	# x_text = positive_examples + negative_examples
	# x_text = [clean_str(sent) for sent in x_text]
	# # Generate labels
	# positive_labels = [[0, 1] for _ in positive_examples]
	# negative_labels = [[1, 0] for _ in negative_examples]
	# y = np.concatenate([positive_labels, negative_labels])

	data = list(open("./data/demo.txt", "r").readlines())
	x_text = []
	y = np.zeros(shape=(len(data), 3))
	labels_counter = 0
	labels = {}

	for i, e in enumerate(data):
		tab_index = e.index('\t')
		label, text = e[0:tab_index], e[tab_index:]
		if label not in labels:
			labels[label] = labels_counter
			labels_counter += 1
		y.itemset(i, labels[label], 1)
		x_text.append(clean_str(text))

	return [x_text, y, labels]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
	"""
	Generates a batch iterator for a dataset.
	"""
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int(len(data) / batch_size) + 1
	for epoch in range(num_epochs):
		# Shuffle the data at each epoch
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data
		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]


def save_labels_to_file(output, labels):
	with open(output + '/labels', 'w') as target:
		target.write(str(labels))


def read_labels(output):
	return eval(open(output + '/labels', 'r').read())
