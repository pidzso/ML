__author__ = 'balazs.pejo'

from collections import Counter

class Clustering(object):

    def __init__(self):
        self.density = 0
        self.users = 0
        self.items = 0
        self.itembased_rates = {}
        self.userbased_rates = {}
        self.normalized_itembased_rates = {}
        self.normalized_userbased_rates = {}
        self.item_rate_number = {}
        self.item_avg_rate = {}
        self.user_rate_number = {}
        self.user_avg_rate = {}

    def get_ratings(self, source):

        self.itembased_rates.clear()
        self.userbased_rates.clear()
        self.users = 0
        self.items = 0

        num = 0
        with open(source) as infile:
            for line in infile:
                l = line.split('\t')

                if int(l[1]) not in self.itembased_rates:
                    self.itembased_rates.update({int(l[1]): {int(l[0]): float(l[2])}})
                else:
                    self.itembased_rates[int(l[1])].update({int(l[0]): float(l[2])})

                if int(l[0]) not in self.userbased_rates:
                    self.userbased_rates.update({int(l[0]): {int(l[1]): float(l[2])}})
                else:
                    self.userbased_rates[int(l[0])].update({int(l[1]): float(l[2])})

                num += 1
                if int(l[1]) > self.items:
                    self.items = int(l[1])
                if int(l[0]) > self.users:
                    self.users = int(l[0])

        self.density = num / (self.users * self.items)

    def normalize(self):
        for user in self.userbased_rates.keys():
            self.user_rate_number.update({user: len(self.userbased_rates[user])})
            self.user_avg_rate.update({user: 0})
        for item in self.itembased_rates.keys():
            self.item_rate_number.update({item: len(self.itembased_rates[item])})
            self.item_avg_rate.update({item: 0})

        for user in self.userbased_rates.keys():
            for item in self.userbased_rates[user].keys():
                self.user_avg_rate[user] += self.userbased_rates[user][item]
            self.user_avg_rate[user] = self.user_avg_rate[user] / self.user_rate_number[user]
        for item in self.itembased_rates.keys():
            for user in self.itembased_rates[item].keys():
                self.item_avg_rate[item] += self.itembased_rates[item][user]
            self.item_avg_rate[item] = self.item_avg_rate[item] / self.item_rate_number[item]

        for user in self.userbased_rates.keys():
            self.normalized_userbased_rates.update({user: {}})
            for item in self.userbased_rates[user].keys():
                self.normalized_userbased_rates[user].update({item: self.userbased_rates[user][item] / self.user_avg_rate[user]})
        for item in self.itembased_rates.keys():
            self.normalized_itembased_rates.update({item: {}})
            for user in self.itembased_rates[item].keys():
                self.normalized_itembased_rates[item].update({user: self.itembased_rates[item][user] / self.item_avg_rate[user]})

    def top(self, k):
        return dict(Counter(self.item_rate_number).most_common(k))
        #return max(self.item_rate_number, key=lambda key: self.item_rate_number[key])

simulate = Clustering()
simulate.get_ratings("100k.data")
simulate.normalize()
print(simulate.top(10))
