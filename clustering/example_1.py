import numpy as np
import collections
import scipy.cluster
import random
import math
# 50, 150, 500, 15

d2_centers = 100
#alpha = 150
#n_coresets = 500
n_iters = 100
random.seed(3)
def mapper(key, value):
	# D2 sampling

	# choose the first point

	curr_center = value[np.random.randint(value.shape[0])]
	centers = []
	centers.append(curr_center)

	#iteratively sample new points
	min_dist = np.full(value.shape[0], np.iinfo(np.int64).max)
	for i in range(d2_centers):
		min_dist = np.minimum(np.apply_along_axis(lambda x: np.linalg.norm(x - curr_center)**2, 1, value), min_dist)
		next_center_idx = np.random.choice(value.shape[0], p=(min_dist / np.sum(min_dist)))
		centers.append(value[next_center_idx])
		curr_center = value[next_center_idx]

	# importance sampling

	# key: data value: the closest center
	#data_cluster_dict = [0] * value.shape[0]
	# key: data value: distance to the closest center
	#data_dist_dict = [0] * value.shape[0]
	# key: center value: data points that belong to the center
	#cluster_data_dict = collections.defaultdict(list)
	#iterate over data and centers to get the three dicts
	#for i in range(value.shape[0]):
	#	d = float("inf")
	#	center_index = 0
	#	for j in range(len(centers)):
	#		new_dist = (np.linalg.norm(value[i] - centers[j], 2) ** 2)
	#		if new_dist < d:
	#			d = new_dist
	#			center_index = j
	#	data_cluster_dict[i] = center_index
	#	data_dist_dict[i] = d
	#	cluster_data_dict[center_index].append(i)
	#steps for importance sampling
	#probs = []
	#c_phi = 1.0 / value.shape[0] * sum(data_dist_dict)
	#for i in range(len(value)):
	#	cluster = data_cluster_dict[i]
	#	B_i = cluster_data_dict[cluster]
	#	first_term = 1.0 * alpha * data_dist_dict[i] / c_phi
	#	second_term = 2.0 * alpha * sum(data_dist_dict[i] for i in B_i) / (c_phi * len(B_i))
	#	third_term = 4.0 * value.shape[0] / len(B_i)
	#	res = first_term + second_term + third_term
	#	probs.append(res)
	#sums = sum(probs)
	#probs = [prob/sums for prob in probs]    
	#importance_sampling_index = np.random.choice(value.shape[0], size = n_coresets, p = probs)
	#coresets = []
	#weights = [1.0 / (probs[i] * n_coresets) for i in importance_sampling_index]
	#for i in importance_sampling_index:
	#	coresets.append(value[i])
	#res = [cores, weights]
	yield 1, [value, centers]
	#yield 1, [coresets, centers]
	#yield 1, coresets

def reducer(key, values):
	X = np.concatenate(([values[i][0] for i in range(values.shape[0])]), axis=0)
	#mu0 = [values[i][1] for i in range(values.shape[0])]
	#random_idx = np.random.randint(0, len(mu0)-1)
	#mu = mu0[random_idx]
	
	mu0 = np.concatenate(([values[i][1] for i in range(values.shape[0])]), axis=0)
	random_idx = np.random.randint(0, len(mu0)-1, size = 200)
	mu = [mu0[i] for i in random_idx]
	#centers = scipy.cluster.vq.kmeans(X, mu, iter = 10)[0]
	centers = scipy.cluster.vq.kmeans(X, mu, iter= n_iters)[0]
	#yield centers
	#k-means with weights from importance sampling(doesn't seem to work)
	#data = [values[i][j] for i in range(values.shape[0]) if i % 2 == 0 for j in range(coresets)]
	#weight = [values[i][j] for i in range(values.shape[0]) if i % 2 == 1 for j in range(coresets)]
	#center_indices = np.random.choice(len(data), 200)
	#centers = [data[i] for i in center_indices]
	#centers = scipy.cluster.vq.kmeans(values, k_or_guess = 200, iter= n_iters)[0]
	#w = [weight[i] for i in center_indices]
	#for i in range(n_iters):
	#	for j in range(len(data)):
	#		dist = []
	#		for k in range(len(centers)):
	#			dist.append(np.linalg.norm(np.array(centers[k])-np.array(data[j]), 2) * w[k])
	#		c = np.argmin(dist)
	#		eta = min(float(j+1), float(c)/(j+1))
	#		centers[c] = centers[c] + eta * (np.array(data[j]) - np.array(centers[c]))
	yield centers
