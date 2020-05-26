import numpy as np

m = 10000
a = 10
sigma = 6

def transform(X):
	np.random.seed(123)
	new_X = []
	w = np.random.multivariate_normal(mean = [0] * X.shape[1], cov = np.identity(X.shape[1]) * sigma**2, size = m)
	b = np.random.uniform(0, 2 * np.pi, m)
	X_new = np.sqrt(2.0 / m) * np.cos(np.dot(X, np.transpose(w)) + b)
	X_new = (X_new - np.mean(X_new, 0)) / np.std(X_new, 0)
	return X_new

def mapper(key, value):
	Y = []
	X = []
	res = []

	for i in range(len(value)):
		feature = value[i].split()[1:]
		label = value[i].split()[0]
		X.append([float(m) for m in feature])
		Y.append(float(label))

	X = np.array(X)
	X = transform(X)
	Y = np.array(Y)

	n_features = len(X[0])
	w = np.zeros(n_features)
	
	for i in range(len(value)):
		x = X[i]
		y = Y[i]
		if y * np.dot(x,w) < 1:
			eta = 1 / np.sqrt(i + 1.0)
			w += [eta * y * xi for xi in x]
			w = w * min(1, 1 / (np.sqrt(a) * np.linalg.norm(w, 2)))
		res.append(w)

	yield 1, np.mean(res, axis=0)


def reducer(key, values):
	yield np.mean(values, axis=0)


