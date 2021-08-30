__author__ = 'balazs.pejo'

import random

class Slope(object):

    def __init__(self):
        self.density = 0
        self.users = 0
        self.items = 0
        self.itembased_rates = {}
        self.userbased_rates = {}
        self.rated_both = {}
        self.deviations = {}
        self.prediction = {}
        self.selected_predictions = {}
        self.selected_deviations = {}

    def get_ratings(self, source):

        self.itembased_rates.clear()
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

    def comp_stage(self):

        self.rated_both.clear()
        self.deviations.clear()

        for item1 in self.itembased_rates:
            self.rated_both.update({item1: {}})
            self.deviations.update({item1: {}})

            for item2 in self.itembased_rates:
                self.rated_both[item1].update({item2: 0})
                self.deviations[item1].update({item2: 0})

                if item1 != item2:

                    for user in self.itembased_rates[item1]:
                        if user in self.itembased_rates[item2]:
                            self.rated_both[item1][item2] += 1
                            self.deviations[item1][item2] += self.itembased_rates[item1][user] - self.itembased_rates[item2][user]

    def select_dev(self, ran):

        self.selected_deviations.clear()
        num = 0
        l = list(self.itembased_rates.keys())

        while num != ran:
            item1 = random.choice(l)
            item2 = random.choice(l)
            if item1 != item2:
                num += 1
                if item1 not in self.selected_deviations:
                    self.selected_deviations.update({item1: {item2: 0}})
                else:
                    self.selected_deviations[item1].update({item2: 0})

    def compute_selected_dev(self):

        self.rated_both.clear()
        self.deviations.clear()

        for item1 in self.selected_deviations:
            self.rated_both.update({item1: {}})
            self.deviations.update({item1: {}})

            for item2 in self.selected_deviations[item1]:
                self.rated_both[item1].update({item2: 0})
                self.deviations[item1].update({item2: 0})

                for user in self.itembased_rates[item1]:
                    if user in self.itembased_rates[item2]:
                        self.rated_both[item1][item2] += 1
                        self.deviations[item1][item2] += self.itembased_rates[item1][user] - self.itembased_rates[item2][user]

    def select_pred(self, ran):

        self.selected_predictions.clear()
        num = 0
        l = list(self.itembased_rates.keys())

        while num != ran:
            item = random.choice(l)
            user = random.randrange(self.users)
            if self.itembased_rates[item].get(user + 1) is None:
                num += 1
                if item not in self.selected_predictions:
                    self.selected_predictions.update({item: {user + 1: 0}})
                else:
                    self.selected_predictions[item].update({user + 1: 0})

    def compute_selected_pred(self):

        for item1 in self.selected_predictions:
            self.prediction.update({item1: {}})
            for user in self.selected_predictions[item1]:
                nom = 0
                denom = 0

                already_rated_items = self.userbased_rates[user]

                delta = self.deviations[item1]
                phi = self.rated_both[item1]

                for item2 in already_rated_items:
                    if item2 == item1:
                        continue

                    both = phi[item2]

                    if both == 0:
                        continue

                    nom += delta[item2] + already_rated_items[item2] * both
                    denom += both

                if denom < 1:
                    self.selected_predictions[item1].update({user: 0})
                else:
                    self.selected_predictions[item1].update({user: nom/denom})

    def predict(self):
        users = self.userbased_rates.keys()
        items = self.itembased_rates.keys()
        self.prediction.clear()

        for user in users:
            self.prediction.update({user: {}})
            already_rated_items = self.userbased_rates[user]
            for item1 in items:
                if item1 in already_rated_items:
                    continue
                nom = 0
                denom = 0

                delta = self.deviations[item1]
                phi = self.rated_both[item1]

                for item2 in already_rated_items:
                    if item2 == item1:
                        continue

                    both = phi[item2]

                    if both == 0:
                        continue

                    nom += delta[item2] + already_rated_items[item2] * both
                    denom += both

                if denom < 1:
                    self.prediction[user].update({item1: 0})
                else:
                    self.prediction[user].update({item1: nom/denom})

    def compute_both_stage(self):

        self.rated_both.clear()
        self.deviations.clear()
        self.prediction.clear()

        for item1 in self.selected_predictions:

            self.rated_both.update({item1: {}})
            self.deviations.update({item1: {}})

            for item2 in self.itembased_rates:
                self.rated_both[item1].update({item2: 0})
                self.deviations[item1].update({item2: 0})

                for user in self.itembased_rates[item2]:
                    if user in self.itembased_rates[item1]:
                        self.rated_both[item1][item2] += 1
                        self.deviations[item1][item2] += self.itembased_rates[item1][user] - self.itembased_rates[item2][user]

        for item1 in self.selected_predictions:

            self.prediction.update({item1: {}})
            for user in self.selected_predictions[item1]:
                nom = 0
                denom = 0

                already_rated_items = self.userbased_rates[user]

                delta = self.deviations[item1]
                phi = self.rated_both[item1]

                for item2 in already_rated_items:
                    if item2 == item1:
                        continue

                    both = phi[item2]

                    if both == 0:
                        continue

                    nom += delta[item2] + already_rated_items[item2] * both
                    denom += both

                if denom < 1:
                    self.selected_predictions[item1].update({user: 0})
                else:
                    self.selected_predictions[item1].update({user: nom/denom})
