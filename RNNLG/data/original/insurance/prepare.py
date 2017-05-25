import json
import numpy as np


def parse(dact):
	acttype = dact.split('(')[0]
	slt2vals = dact.split('(')[1].replace(')', '').split(';')
	jsact = {'acttype': acttype, 's2v': []}
	for slt2val in slt2vals:
		if slt2val == '':  # no slot
			jsact['s2v'].append((None, None))
		elif '=' not in slt2val:  # no value
			slt2val = slt2val.replace('_', '').replace(' ', '')
			jsact['s2v'].append((slt2val.strip('\'\"'), '?'))
		else:  # both slot and value exist
			split = slt2val.split('=')
			keep_values = '\'' not in split[1] and '\"' not in split
			s, v = [x.strip('\'\"') for x in split]
			lexicalised_val = v
			s = s.replace('_', '').replace(' ', '')
			for key, vals in special_values.iteritems():
				if v in vals:  # unify the special values
					v = key
			if not special_values.has_key(v) and not keep_values:
				lexicalised_val = '_'  # delexicalisation
			jsact['s2v'].append((s, lexicalised_val, v, keep_values))
	print jsact
	return jsact


# read data
data = json.load(file('all.json'))

# create test split
data_split = int(len(data) * 0.25)
np.random.shuffle(data)
train = data[data_split:]
test = data[:data_split]

# create train and test file
json.dump(train, open("train.json", "wb"), indent=2)
json.dump(test, open("test.json", "wb"), indent=2)
json.dump(test, open("valid.json", "wb"), indent=2)
print "Size of training set: %d" % len(train)
print "Size of test set: %d" % len(test)

# prepare dictionaries
detect_pairs = {'general': {}}
special_slots = json.load(file('../../../resource/special_slots.txt'))
special_values = json.load(file('../../../resource/special_values.txt'))
features = set()
vocab = set()

for d in data:
	dact = parse(d[0])
	features.add('a.' + dact['acttype'])
	slot_counter = {}

	# go through the slots for a dialog
	for slot in dact['s2v']:
		name = 's.' + slot[0]
		keep_value = slot[3]

		# update the counter
		slot_counter[name] = slot_counter.get(name, 0) + 1

		# add the slot
		features.add(name)
		if keep_value:  # todo not sure if this is right or necessary
			special_slots.append(slot[0])
		detect_pairs['general'][slot[0]] = 'SLOT_' + str(slot[0]).upper()
		vocab.add(detect_pairs['general'][slot[0]])

		# add the slot value
		val = slot[1] if keep_value else slot[1] + str(slot_counter[name])
		sv = 'sv.' + slot[0] + '.' + val
		features.add(sv)

	for i in range(1, len(d)):
		for w in d[i].split(' '):
			vocab.add(w)

print '\n\nDetect Pairs'
for s, alias in detect_pairs['general'].items():
	print '"%s" : "%s"  ' % (s, alias)

print '\n\nFeatures Template'
for f in sorted(features):
	print f

print '\n\nSpecial Slots'
for s in set(special_slots):
	print s

print '\n\nVocab'
for v in sorted(vocab):
	print v
