#!/usr/bin/python3

import random


def conversation_2_seqs(conv):
	sequences = []
	sequence = ""
	for utt in conv:
		if utt.speaker == "user" and len(sequence) > 0:
			sequences.append((sequence.strip(), utt.content))
		elif utt.speaker == "agent" and len(sequence) > 0 and len(sequences) == 0:
			sequences.append((sequence.strip(), utt.content))
		sequence += utt.content + " "
	return sequences


def read_conversations(convs_path):
	convs = []
	with open(convs_path, 'r') as file:
		conversation = []
		for line in file:
			if line.strip() == '<EOC>':
				convs.append(conversation)
				conversation = []
				continue
			speaker_end_index = line.index(':')
			speaker, content = line[0:speaker_end_index], line[speaker_end_index + 1:]
			conversation.append(Utterance(speaker.strip(), content.strip().lower()))
	print('There are ' + str(len(convs)) + ' conversations')
	return convs


class Utterance:
	def __init__(self, speaker, content):
		self.speaker = speaker
		self.content = content


def read_conversations_2_seqs(convs_path):
	conversations = read_conversations(convs_path)
	sequences = {}
	for i, c in enumerate(conversations):
		for seq in conversation_2_seqs(c):
			if i not in sequences:
				sequences[i] = []
			sequences[i].append(seq)
	return sequences


def write_sequences_to_file(file, sequences, index):
	for s in sequences:
		file.write(s[index] + '\n')


def main():
	conversations_file = 'conversations.txt'
	test_percentage = 0.20
	path = ''

	# read the conversations into sequence tuples
	sequences = read_conversations_2_seqs(conversations_file)

	# sample test ids
	test_size = int(len(sequences) * test_percentage)
	test_ids = random.sample([i for i in range(len(sequences))], test_size)

	# open files
	train_enc = open(path + 'train.enc', 'w')
	train_dec = open(path + 'train.dec', 'w')
	test_enc = open(path + 'test.enc', 'w')
	test_dec = open(path + 'test.dec', 'w')

	for i in range(len(sequences)):
		if i in test_ids:
			write_sequences_to_file(test_enc, sequences[i], 0)
			write_sequences_to_file(test_dec, sequences[i], 1)
		else:
			write_sequences_to_file(train_enc, sequences[i], 0)
			write_sequences_to_file(train_dec, sequences[i], 1)


	# close files
	train_enc.close()
	train_dec.close()
	test_enc.close()
	test_dec.close()


if __name__ == "__main__":
	main()
