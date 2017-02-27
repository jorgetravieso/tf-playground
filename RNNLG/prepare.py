from loader.DataReader import DataReader
import json


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


data = json.load(file('all.json'))

detect_pairs = {'general': {}}
special_slots = json.load(file('resource/special_slots.txt'))
special_values = json.load(file('resource/special_values.txt'))
features = []
vocab = set()

for d in data:
	dact = parse(d[0])
	features.append('a.' + dact['acttype'])
	slot_counter = {}

	# go through the slots for a dialog
	for slot in dact['s2v']:
		name = 's.' + slot[0]
		keep_value = slot[3]

		# update the counter
		slot_counter[name] = slot_counter.get(name, 0) + 1

		# add the slot
		features.append(name)
		if keep_value:  # todo not sure if this is right or necessary
			special_slots.append(slot[0])
		detect_pairs['general'][slot[0]] = 'SLOT_' + str(slot[0]).upper()

		# add the slot value
		val = slot[1] if keep_value else slot[1] + str(slot_counter[name])
		sv = 'sv.' + slot[0] + '.' + val
		features.append(sv)

	for i in range(1, len(d)):
		for w in d[i].split(' '):
			vocab.add(w)

print '\n\nDetect Pairs'
for s, alias in detect_pairs['general'].items():
	print s + ':' + str(alias)

print '\n\nFeatures Template'
for f in sorted(features):
	print f

print '\n\nVocab'
for v in sorted(vocab):
	print v
