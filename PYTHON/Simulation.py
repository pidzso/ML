__author__ = 'balazs.pejo'

from WeightedSlopeOne import Slope
from SGD import SGD
from time import time
from sys import argv

dataIN = argv[1]
rateIN = argv[2]

dev = {"10m": {}, "1m": {}}
both = {"10m": {}, "1m": {}}

def run_weighted_slope(data, rat):
    
    rat = float(rat)
    
    with open(str("OUT_" + dataIN + "_" + rateIN), "a") as myfile:
        myfile.write(str("Dataset:\t" + data + "\n"))
    with open(str("OUT_" + dataIN + "_" + rateIN), "a") as myfile:
        myfile.write("Ratio:\t" + str(rat) + "\n")
    simulate = Slope()

    with open(str("OUT_" + dataIN + "_" + rateIN), "a") as myfile:
        myfile.write("Reading ...\n")
    start = time()
    simulate.get_ratings(data + ".data")
    stop = time()
    with open(str("OUT_" + dataIN + "_" + rateIN), "a") as myfile:
        myfile.write(str(stop - start) + " sec\n")

    if rat == 1:

        with open(str("OUT_" + dataIN + "_" + rateIN), "a") as myfile:
            myfile.write("Computational stage ...\n")
        start = time()
        simulate.comp_stage()
        stop = time()
        with open(str("OUT_" + dataIN + "_" + rateIN), "a") as myfile:
            myfile.write(str(stop - start) + " sec\n")

        with open(str("OUT_" + dataIN + "_" + rateIN), "a") as myfile:
            myfile.write("Prediction stage ...\n")
        start = time()
        simulate.predict()
        stop = time()
        with open(str("OUT_" + dataIN + "_" + rateIN), "a") as myfile:
            myfile.write(str(stop - start) + " sec\n")

    else:

        dev_n = round((simulate.items * simulate.items - simulate.items) / 2 * rat)
        pred_n = round((1 - simulate.density) * (simulate.items * simulate.users) * rat)
        with open(str("OUT_" + dataIN + "_" + rateIN), "a") as myfile:
            myfile.write("Calculating both stage ...\n")
        start = time()
        simulate.select_pred(pred_n)
        simulate.compute_both_stage()
        stop = time()
        with open(str("OUT_" + dataIN + "_" + rateIN), "a") as myfile:
            myfile.write(str(stop - start) + " sec\n")

        with open(str("OUT_" + dataIN + "_" + rateIN), "a") as myfile:
            myfile.write("Calculating comp stage ...\n")
        start = time()
        simulate.select_dev(dev_n)
        simulate.compute_selected_dev()
        stop = time()
        with open(str("OUT_" + dataIN + "_" + rateIN), "a") as myfile:
            myfile.write(str(stop - start) + " sec\n")

        with open(str("OUT_" + dataIN + "_" + rateIN), "a") as myfile:
            myfile.write("Auxiliary computation for prediction stage (comp stage) ...\n")
        if dev[data] == {}:
            simulate.comp_stage()
            dev[data] = simulate.deviations
            both[data] = simulate.rated_both

        simulate.deviations = dev[data]
        simulate.rated_both = both[data]

        with open(str("OUT_" + dataIN + "_" + rateIN), "a") as myfile:
            myfile.write("Calculating pred stage ...\n")
        start = time()
        simulate.select_pred(pred_n)
        simulate.compute_selected_pred()
        stop = time()
        with open(str("OUT_" + dataIN + "_" + rateIN), "a") as myfile:
            myfile.write(str(stop - start) + " sec\n")


def run_sgd(source, steps, learn_rate, reg_rate):

    simulate = SGD()
    simulate.get_rates(source)
    start = time()
    simulate.mf(steps, learn_rate, reg_rate)
    stop = time()
    with open(str("OUT_" + dataIN + "_" + rateIN), "a") as myfile:
        myfile.write("All:\t" + str(stop - start) + "\n")


run_weighted_slope(dataIN, rateIN)
#run_sgd("10m.data", 100, 0.0005, 0.0001)
