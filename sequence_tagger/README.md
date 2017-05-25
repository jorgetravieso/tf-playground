## Sequence Tagging with TensorFlow

#### Requirements
Python 2.7
TensorFlow 1.0

#### References
[1] https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html
[2] https://arxiv.org/pdf/1603.01354.pdf

#### Task
Given a sentence, give a tag to each word. A classical application is Named Entity Recognition (NER). Here is an example

```
John lives in New York
PER  O     O  LOC LOC
```

#### Model

Similar to [this paper](https://arxiv.org/pdf/1603.01354.pdf).

- concatenate final states of a bi-lstm on character embeddings to get a character-based representation of each word
- concatenate this representation to a standard word vector representation (GloVe here)
- run a bi-lstm on each sentence to extract contextual representation of each word
- decode with a linear chain CRF

#### Data

The training data must be in the following format (identical to the CoNLL2003 dataset)

```
John B-PER
lives O
in O
New B-LOC
York I-LOC
. O

This O
is O
another O
sentence O
```


#### Usage

First, build vocab from the data and extract trimmed glove vectors according to the config in `config.py`.

```
python prepare.py
```

Second, train and test model with 

```
python main.py
```

data iterators and utils are in `data_utils.py` and the model with training/test procedures are in `model.py`


