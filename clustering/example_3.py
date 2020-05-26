import numpy as np
import math
import collections

''' CONSTANTS '''
num_d_squared_centers = 30
mapper_coreset_size = 500 - num_d_squared_centers
k = 200
alpha = 20 * (math.log(k, 2) + 1)
k_means_iterations = 15


# key: None
# value: 2d numpy array
def mapper(key, value):
    centers = list()

    ''' D^2 sampling '''

    curr_center = value[np.random.randint(value.shape[0])]
    centers.append(curr_center)

    # initialization of minimum squared distances
    min_squared_distances = np.full(value.shape[0], np.iinfo(np.int64).max)

    # gets the squared distance to the current center from a specified vector
    def get_squared_dist_to_curr_center(vec):
        return np.linalg.norm(vec - curr_center)**2

    # generate the centers
    for _ in xrange(num_d_squared_centers - 1):
        min_squared_distances = np.minimum(np.apply_along_axis(get_squared_dist_to_curr_center, 1, value), min_squared_distances)
        next_center_idx = np.random.choice(value.shape[0], p=(min_squared_distances / np.sum(min_squared_distances)))
        centers.append(value[next_center_idx])
        curr_center = value[next_center_idx]

    ''' significance sampling '''

    # gets the squared distance to the nearest center from a given vector
    def squared_distance_to_nearest_center(vec):
        return get_distance_to_nearest_center(vec, centers)**2

    # sum of distances from each point to its nearest D^2 center
    distances_sum = np.sum(np.apply_along_axis(squared_distance_to_nearest_center, 1, value))
    # map from each center to a list of the distances of the points that are closest to that center
    centers_to_nearest_squared_dists = get_centers_to_nearest_point_squared_dists(value, centers)

    def significance_sampling_prob(vec):
        ave_nearest_dist = (1.0 / value.shape[0]) * distances_sum
        nearest_center, squared_dist_to_nearest_center = get_nearest_center_and_squared_dist_tuple(vec, centers)
        squared_dists_to_nearest_center = centers_to_nearest_squared_dists[nearest_center.tostring()]
        first_term = alpha * squared_dist_to_nearest_center / ave_nearest_dist
        second_term = 2.0 * alpha * sum(squared_dists_to_nearest_center) / (len(squared_dists_to_nearest_center) * ave_nearest_dist)
        third_term = 4.0 * value.shape[0] / len(squared_dists_to_nearest_center)
        return first_term + second_term + third_term




    sig_sampling_vals = np.apply_along_axis(significance_sampling_prob, 1, value)
    sig_sampling_probs = sig_sampling_vals / np.sum(sig_sampling_vals)
    sig_sampled_indices = np.random.choice(value.shape[0], size=mapper_coreset_size, p=sig_sampling_probs)

    for index in sig_sampled_indices:
        centers.append(value[index])

    yield 0, centers


# key: key from mapper used to aggregate
# values: list of all value for that key
# Note that we do *not* output a (key, value) pair here.
def reducer(key, values):
    centers = list()
    center_indices = np.random.choice(len(values), k)

    for idx in center_indices:
        centers.append(values[idx])

    for i in xrange(1, k_means_iterations + 1):
        centers_to_indices = get_centers_to_nearest_point_indices(values, centers)
        new_centers = list()

        for center in centers_to_indices.keys():
            vec_sum = np.zeros(values.shape[1])
            for idx in centers_to_indices[center]:
                vec_sum = vec_sum + values[idx]

            mean_vec = vec_sum / len(centers_to_indices[center])
            new_centers.append(mean_vec)

        # in the case that some centers we picked were outliers
        if len(new_centers) < k:
            for j in xrange(k - len(new_centers)):
                new_centers.append(values[np.random.randint(values.shape[0])])

        centers = new_centers

    yield centers


''' helper functions '''


# Given a vector and a list of vectors that are the current centers, return
# the distance to the closest center.
def get_distance_to_nearest_center(vec, centers):
    min_dist = np.iinfo(np.int64).max
    for center in centers:
        dist = np.linalg.norm(vec - center)
    if dist < min_dist:
        min_dist = dist

    return min_dist


# Given a 2d numpy array and a list of vectors that are the current centers, return a map
# from each center to a list of scalars that represents the distances to the points in arr2d
# closest to that center.
def get_centers_to_nearest_point_squared_dists(arr2d, centers):
    ret = collections.defaultdict(list)
    for i in range(arr2d.shape[0]):
        center, dist = get_nearest_center_and_squared_dist_tuple(arr2d[i], centers)
        ret[center.tostring()].append(dist)

    return ret


# Given a vector and a list of vectors that are the current centers, return the center
# that is closest to the specified vector.
def vec_to_nearest_center(vec, centers):
    min_dist = np.iinfo(np.int32).max
    nearest_center = None
    for center in centers:
        dist = np.linalg.norm(vec - center)
        if dist < min_dist:
            min_dist = dist
            nearest_center = center

    return nearest_center


# Given a 2d numpy array and a list of vectors that are the current centers, return a
# map from each center to a list of indices that represents the indices of the vectors
# arr2d closest to that center.
def get_centers_to_nearest_point_indices(arr2d, centers):
    ret = collections.defaultdict(list)
    for j in range(arr2d.shape[0]):
        curr_center = vec_to_nearest_center(arr2d[j], centers)
        ret[curr_center.tostring()].append(j)

    return ret


# Given a vector and a list of vectors that are the current centers, return a tuple
# of (nearest_center, squared_min_dist).
def get_nearest_center_and_squared_dist_tuple(vec, centers):
    min_dist = np.iinfo(np.int32).max
    ret_center = None
    for center in centers:
        dist = np.linalg.norm(vec - center)
        if dist < min_dist:
            min_dist = dist
            ret_center = center

    return ret_center, min_dist**2