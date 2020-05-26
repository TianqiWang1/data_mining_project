import numpy as np
import scipy.cluster

k = 200
n_features = 250
#n_iter = 100
alpha = 10.0
n_points = 500

def dist_from_center(X, mu):
    D2 = np.array([np.linalg.norm(x-mu)**2 for x in X])
    return D2

def dist_from_centers(X, mu):
    D2 = np.array([{index: np.linalg.norm(x-c)**2 for index,c in enumerate(mu)} for x in X])
    return D2

def choose_next_point(X, probs):
    cumprobs = probs.cumsum()
    cumprobs.sort()
    r = np.random.random()
    ind = np.where(cumprobs >= r)[0][0]
    return X[ind]

def init_centers(X):
    mu = [np.random.randn(n_features)]
    D2 = dist_from_center(X, mu)

    for i in range(k):
        newD2 = dist_from_center(X, mu[i])
        for j in range(len(X)):
            if newD2[j] < D2[j]:
                D2[j] = newD2[j]
        probs = D2 / D2.sum()
        mu.append(choose_next_point(X, probs))
    return mu

def mapper(key, value):
    # key: None
    # value: one line of input file
    # --------------------------------------------------------------------------
    X = value
    mu = np.array(init_centers(X))

    # Importance sampling

    D2 = dist_from_centers(X, mu)
    D2_value = np.array([min(D2[i].values()) for i in range(D2.shape[0])])
    key_of_closest_center = [min(D2[i].items(), key=(lambda x: x[1]))[0] for i in range(D2.shape[0])]
    index_list = []
    for l in range(len(D2[0])):
        index_list.append([index for index, value in enumerate(key_of_closest_center) if value == l])

    sum_dist_Bi_dict = []
    n_Bi_dict = []
    for m in range(len(D2[0])):
        dist_Bi = []
        for n in index_list[m]:
            dist_Bi.append(D2_value[n])
        sum_dist_Bi_dict.append({i: sum(dist_Bi) for i in index_list[m]})
        n_Bi_dict.append({i: len(index_list[m]) for i in index_list[m]})

    sum_dist_Bi = np.zeros(D2.shape[0])
    for m in range(len(D2[0])):
        for key in sum_dist_Bi_dict[m].keys():
            sum_dist_Bi[int(key)] = sum_dist_Bi_dict[m][key]

    n_Bi = np.zeros(D2.shape[0])
    for m in range(len(D2[0])):
        for key in n_Bi_dict[m].keys():
            n_Bi[int(key)] = n_Bi_dict[m][key]

    c = D2_value.sum()/len(X)
    q = (alpha*D2_value)/c + 2.0*alpha*sum_dist_Bi/(n_Bi*c) + 4.0*len(X)/n_Bi
    q = q/sum(q)
    # print q

    coreset = []
    
    while len(coreset) < n_points: # n_points: number of points in a coreset
        coreset.append(choose_next_point(X, q))
    coreset = np.array(coreset)
    yield 1, coreset
    #yield "key", [coreset, mu]  # this is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    # --------------------------------------------------------------------------
    # initialization of centers
    mu0 = values[0][1]
    X = np.concatenate(([values[i][0] for i in range(values.shape[0])]), axis=0)#

#    n_samples = len(X)
#    index = np.random.permutation(n_samples)
#    for t in range(1, n_iter):
#        xt = X[index[t]]
#        dist = []
#        for i in range(k):
#            dist.append(np.linalg.norm(xt - mu[i]))
#        c = dist.index(min(dist))
#        eta = float(c) / float(t)
#        mu[c] = mu[c] + eta * (xt - mu[c])

    mu = scipy.cluster.vq.kmeans(X, mu0)[0]
    yield mu
