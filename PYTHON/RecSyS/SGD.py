import numpy as np
from time import time


class SGD(object):

    def __init__(self):

        self.users = 71567
        self.items = 10681
        self.samples = 10000054
        self.features = 10
        self.rates = np.zeros((self.users, self.items))

        np.random.seed(0)
        self.user_features = np.random.rand(self.users, self.features)
        self.item_features = np.random.rand(self.features, self.items)
        self.iter_uf = np.zeros((101, self.users, self.features))
        self.iter_if = np.zeros((101, self.features, self.items))

        self.chosen_features_u = np.zeros((101, self.users, self.features))
        self.chosen_features_i = np.zeros((101, self.features, self.items))

    def chose(self, part):

        np.random.seed(0)
        r = np.random.random()

        self.chosen_features_u = np.array([0] * round((1 - r * part) * (11 * self.users * self.features)) +
                                          [1] * round(r * part * (11 * self.users * self.features)))
        self.chosen_features_i = np.array([0] * round((1 - (1-r) * part) * (11 * self.items * self.features)) +
                                          [1] * round((1-r) * part * (11 * self.items * self.features)))
        np.random.shuffle(self.chosen_features_u)
        np.random.shuffle(self.chosen_features_i)

        self.chosen_features_u.reshape(11, self.users, self.features)
        self.chosen_features_i.reshape(11, self.features, self.items)

    def get_rates(self, source):

        with open(source) as infile:
            for line in infile:
                l = line.split('\t')
                self.rates[int(l[0])-1][int(l[1])-1] = float(l[2])

    def mf(self, steps, learning, regularization):

        np.random.seed(0)
        self.iter_uf[0] = np.random.rand(self.users, self.features)
        self.iter_if[0] = np.random.rand(self.features, self.items)
        np.random.seed(0)

        iter_t = 0
        upd_t = 0

        for step in range(steps):
            print("Step:\t", step + 1)

            self.iter_uf[step+1] = self.user_features
            self.iter_if[step+1] = self.item_features

            start0 = time()

            for user in np.random.permutation(range(len(self.rates))):
                for item in np.random.permutation(range(len(self.rates[user]))):

                    start1 = time()

                    if self.rates[user][item] > 0:

                        error_ui = self.rates[user][item] - np.dot(self.user_features[user, :],
                                                                   self.item_features[:, item])
                        for feature in range(self.features):
                            aux = self.user_features[user][feature] + learning * (
                                  2 * error_ui * self.item_features[feature][item] -
                                  regularization * self.user_features[user][feature])
                            self.item_features[feature][item] = self.item_features[feature][item] + learning * (
                                                                2 * error_ui * self.user_features[user][feature] -
                                                                regularization * self.item_features[feature][item])
                            self.user_features[user][feature] = aux
                        stop1 = time()
                        upd_t += stop1 - start1

            stop0 = time()
            iter_t += (stop0 - start0) / 100

            if step % 10 == 9:

                error = 0

                for user in range(len(self.rates)):
                    for item in range(len(self.rates[user])):

                        if self.rates[user][item] > 0:
                            error += pow(self.rates[user][item] - np.dot(self.user_features[user, :],
                                                                         self.item_features[:, item]), 2)
                            for feature in range(self.features):
                                error += (regularization / 2) * (pow(self.user_features[user][feature], 2) +
                                                                 pow(self.item_features[feature][item], 2))
                print("RMSE", step + 1, ":", np.sqrt(error/self.samples))

        print("Iter:\t", iter_t)
        print("Upd:\t", upd_t / 1000005400)
