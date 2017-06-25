import tensorflow as tf
import numpy as np
import os
import data_utils

learn = tf.contrib.learn


class DataReader:

	def __init__(self, runs_dir, tsv_file, split_factor=0.2):
		# Load data
		x_text, y, labels = data_utils.load_data_and_labels(tsv_file)

		# Build vocabulary
		max_document_length = max([len(x.split(" ")) for x in x_text])
		vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
		x = np.array(list(vocab_processor.fit_transform(x_text)))

		# Randomly shuffle data
		np.random.seed(10)
		shuffle_indices = np.random.permutation(np.arange(len(y)))
		x_shuffled = x[shuffle_indices]
		y_shuffled = y[shuffle_indices]

		# Split train/test set
		x_split_index = int(len(x_shuffled) * split_factor)
		y_split_index = int(len(y_shuffled) * split_factor)
		print("x_shuffled", x_shuffled)

		x_train, x_dev = x_shuffled[:-x_split_index], x_shuffled[-x_split_index:]
		y_train, y_dev = y_shuffled[:-y_split_index], y_shuffled[-y_split_index:]

		self.train = Data(x_train, y_train)
		self.dev = Data(x_dev, y_dev)
		self.max_document_length = max_document_length
		self.num_classes = len(set(labels))

		# Write vocabulary and labels
		vocab_processor.save(os.path.join(runs_dir, "vocab"))
		data_utils.save_labels_to_file(runs_dir, labels)



class Data:
	iter = 0
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.n = len(x)

	def next_batch(self, size):
		batch_x = []
		batch_y = []
		while size > 0:
			batch_x.append(self.x[self.iter])
			batch_y.append(self.y[self.iter])
			self.iter = (self.iter + 1) % self.n
			size -= 1
		return np.array(batch_x), np.array(batch_y)


