import numpy as np
def softmax(x):
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()
print(softmax([0.761,1.765]))
