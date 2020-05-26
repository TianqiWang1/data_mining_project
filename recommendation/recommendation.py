import numpy as np

d = 6
alpha = 4
r1 = 1
r2 = -8


def set_articles(articles):
    global index_all
    global Aa
    global Aa_inv
    global ba
    global theta

    n_articles = len(articles)
    index_all = {}
    Aa = np.zeros((n_articles, d, d))
    Aa_inv = np.zeros((n_articles, d, d))
    ba = np.zeros((n_articles, d, 1))
    theta = np.zeros((n_articles, d, 1))
    i = 0
    for key in articles:
        index_all[key] = i
        Aa[i] = np.identity(d)
        Aa_inv[i] = np.identity(d)
        ba[i] = np.zeros((d, 1))
        theta[i] = np.zeros((d, 1))
        i += 1
	pass


def update(reward):
    if reward == -1:
        return
    elif reward == 1:
        r = r1
    elif reward == 0:
        r = r2

    Aa[max_a] += np.outer(x,x)
    Aa_inv[max_a] = np.linalg.inv(Aa[max_a])
    # ba[max_a] += r * reward * x
    ba[max_a] += r * x
    theta[max_a] = Aa_inv[max_a].dot(ba[max_a])


def recommend(time, user_features, choices):
    global max_a
    global x

    article_len = len(choices)

    x = np.array(user_features).reshape((d,1))
    x_t = np.transpose(x)
    index = [index_all[article] for article in choices]
    UCB = np.transpose(theta[index], (0,2,1)).dot(x).reshape(article_len,1) + alpha * np.sqrt(np.transpose(x).dot(Aa_inv[index]).dot(x)).reshape((article_len,1))

    max_index = np.argmax(UCB)
    max_a = index[max_index]
    return choices[max_index]
