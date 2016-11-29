
def concat_conversation(conv):
	return ''.join((utt + ' ') for utt in conv)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))) # fixme remove
import tensorflow as tf
import execute

print('Starting a new tensorflow session ...')
sess = tf.Session()
sess, model, enc_vocab, rev_dec_vocab = execute.init_session(sess, conf='seq2seq_serve.ini')


while True:
	conversation = []
	print('Jorge: ', end='')
	line = sys.stdin.readline()
	if line == '...':
		print('New Conversation ...\n')
		conversation = []
		continue
	conversation.append(line)
	context = concat_conversation(conversation)
	output = execute.decode_line(sess, model, enc_vocab, rev_dec_vocab, context)
	print('Lisa: ' + output)
	conversation.append(output)

