import numpy as np
import wget
import os.path

embeddings_file = 'data/glove.6B/glove.6B.300d.txt'
if not os.path.isfile(embeddings_file):
	os.makedirs('data/glove.6B/')
	embeddings_url = 'http://learning-resources.jorgetravieso.com/embeddings/glove/glove.6B.300d.txt'
	wget.download(embeddings_url, out=embeddings_file)

from config import config
from data_utils import UNK, NUM, \
	get_glove_vocab, write_vocab, load_vocab, \
	export_trimmed_glove_vectors

vocab_words = set()
vocab_tags = set()
vocab_chars = set()
file = open('data/all.txt')
for line in file:
	line = line.strip()
	if len(line) == 0:
		continue
	token, tag = line.split(' ')
	print token, tag
	for c in token:
		vocab_chars.add(c)
	vocab_words.add(token)
	vocab_tags.add(tag)

# Build Word and Tag vocab
vocab_glove = get_glove_vocab(config.glove_filename)

vocab = vocab_words & vocab_glove
vocab.add(UNK)
vocab.add(NUM)

# Save vocabs
write_vocab(vocab, config.words_filename)
write_vocab(vocab_tags, config.tags_filename)
write_vocab(vocab_chars, config.chars_filename)

# Trim GloVe Vectors
vocab = load_vocab(config.words_filename)
export_trimmed_glove_vectors(vocab, config.glove_filename, config.trimmed_filename, config.dim)