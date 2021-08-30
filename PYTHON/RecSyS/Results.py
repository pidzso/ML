__author__ = 'balazs.pejo'

import numpy  as np
from numpy import concatenate as conc
import matplotlib.pyplot as plt     # plt.imshow()


# number of executions
paralelEX  = 3
# privacy parameters: 0,0.1,0.2,0.3,0.4,0.5
privparam  = np.arange(0, 0.6, 0.1)

# data from the execution self.(D1-D2/Dn2-Dn2)(player number)(privacy method)
real1Sup   = np.zeros(shape=(paralelEX, privparam.size, privparam.size))
real1DP    = np.zeros(shape=(paralelEX, privparam.size, privparam.size))
real2Sup   = np.zeros(shape=(paralelEX, privparam.size, privparam.size))
real2DP    = np.zeros(shape=(paralelEX, privparam.size, privparam.size))
approx1Sup = np.zeros(shape=(paralelEX, privparam.size, privparam.size))
approx1DP  = np.zeros(shape=(paralelEX, privparam.size, privparam.size))
approx2Sup = np.zeros(shape=(paralelEX, privparam.size, privparam.size))
approx2DP  = np.zeros(shape=(paralelEX, privparam.size, privparam.size))

# mean of the execution data
avgR1Sup = np.mean(real1Sup, axis=0)
avgR1DP  = np.mean(real1DP,  axis=0)
avgR2Sup = np.mean(real2Sup, axis=0)
avgR2DP  = np.mean(real2DP,  axis=0)
avgA1Sup = np.mean(approx1Sup, axis=0)
avgA1DP  = np.mean(approx1DP,  axis=0)
avgA2Sup = np.mean(approx2Sup, axis=0)
avgA2DP  = np.mean(approx2DP,  axis=0)


# NF: 'out/NF' + source = dataset
# MLxx: 'out/' + source = dataset
def get_result(source):
    # data format:
    # 1-3:   real1Sup 1-3 col
    # 4-6:   real1Sup 4-6 col
    # 7-9:   real1DP  1-3 col
    # 10-12: real1DP  4-6 col
    # 13-15: real2Sup 1-3 col
    # 16-18: real2Sup 4-6 col
    # ...
    # 93-95: apr2DP   1-3 col
    # 94-96: apr2DP   4-6 col

    for i in np.arange(paralelEX):
        real1Sup[i] = conc((np.vsplit(np.loadtxt('out/NF' + str(source) + '_' + str(i+1) + '.txt'), 16)[0],
                            np.vsplit(np.loadtxt('out/NF' + str(source) + '_' + str(i+1) + '.txt'), 16)[1]), axis=1)
        real1DP[i]  = conc((np.vsplit(np.loadtxt('out/NF' + str(source) + '_' + str(i+1) + '.txt'), 16)[2],
                            np.vsplit(np.loadtxt('out/NF' + str(source) + '_' + str(i+1) + '.txt'), 16)[3]), axis=1)
        real2Sup[i] = conc((np.vsplit(np.loadtxt('out/NF' + str(source) + '_' + str(i+1) + '.txt'), 16)[4],
                            np.vsplit(np.loadtxt('out/NF' + str(source) + '_' + str(i+1) + '.txt'), 16)[5]), axis=1)
        real2DP[i]  = conc((np.vsplit(np.loadtxt('out/NF' + str(source) + '_' + str(i+1) + '.txt'), 16)[6],
                            np.vsplit(np.loadtxt('out/NF' + str(source) + '_' + str(i+1) + '.txt'), 16)[7]), axis=1)
        approx1Sup[i] = conc((np.vsplit(np.loadtxt('out/NF' + str(source) + '_' + str(i+1) + '.txt'), 16)[8],
                              np.vsplit(np.loadtxt('out/NF' + str(source) + '_' + str(i+1) + '.txt'), 16)[9]), axis=1)
        approx1DP[i]  = conc((np.vsplit(np.loadtxt('out/NF' + str(source) + '_' + str(i+1) + '.txt'), 16)[10],
                              np.vsplit(np.loadtxt('out/NF' + str(source) + '_' + str(i+1) + '.txt'), 16)[11]), axis=1)
        approx2Sup[i] = conc((np.vsplit(np.loadtxt('out/NF' + str(source) + '_' + str(i+1) + '.txt'), 16)[12],
                              np.vsplit(np.loadtxt('out/NF' + str(source) + '_' + str(i+1) + '.txt'), 16)[13]), axis=1)
        approx2DP[i]  = conc((np.vsplit(np.loadtxt('out/NF' + str(source) + '_' + str(i+1) + '.txt'), 16)[14],
                              np.vsplit(np.loadtxt('out/NF' + str(source) + '_' + str(i+1) + '.txt'), 16)[15]), axis=1)

    global avgR1Sup
    global avgR1DP
    global avgR2Sup
    global avgR2DP
    global avgA1Sup
    global avgA1DP
    global avgA2Sup
    global avgA2DP
    avgR1Sup = np.mean(real1Sup, axis=0)
    avgR1DP  = np.mean(real1DP,  axis=0)
    avgR2Sup = np.mean(real2Sup, axis=0)
    avgR2DP  = np.mean(real2DP,  axis=0)
    avgA1Sup = np.mean(approx1Sup, axis=0)
    avgA1DP  = np.mean(approx1DP,  axis=0)
    avgA2Sup = np.mean(approx2Sup, axis=0)
    avgA2DP  = np.mean(approx2DP,  axis=0)


# approximation boosting with linear DP
def funcLIN(p1, p2, data, method, phi_SD):

    # f1 optimized methodwise for (a,b) in a+data/b
    # f2 optimized methodwise for (a,b,c) in a+b*p1-c*p2
    if method == "Sup":
        f2 = 0.65 + data / 250
        f1 = 0.92 + 0.52 * p1/10 - 0.33 * p2/10
        # based on monotonizer(data)
        #f2 = 0.64 + data / 200
        #f1 = 0.93 + 0.47 * p1/10 - 0.33 * p2/10
    if method == "DP":
        f2 = 0.70 + data / 150
        f1 = 0.80 + 0.60 * p1/10 - 0.15 * p2/10
        # based on monotonizer(data)
        #f2 = 0.70 + data / 150
        #f1 = 0.80 + 0.56 * p1/10 - 0.12 * p2/10
    return f1 * f2 * phi_SD


# approximation boosting with quardratic DP
def funcQUAD(p1, p2, data, method, phi_SD):

    if method == "Sup":
        f2 = 0.65 + data / 250
        f1 = 0.92 + 0.52 * p1/10 - 0.33 * p2/10
    if method == "DP":
        f2 = 0.70 + data / 150
        f1 = 0.77 + 1.58 * (np.power(p1/10, 2) + np.power(p2/10, 2)) - 0.02 * p1 * p2
        # based on monotonizer(data)
        #f1 = 0.76 + 1.60 * (np.power(p1/10, 2) + np.power(p2/10, 2)) - 0.02 * p1 * p2
    return f1 * f2 * phi_SD


# datasize wise boosting
def func_data(data, method, phi_SD, a, b):

    if method == "Sup":
        x = a + data / b
    if method == "DP":
        x = a + data / b
    return x * phi_SD


# privacy parameter wise boosting
def func_priv(p1, p2, data, method, phi_SD, a, b, c, d):

    if method == "Sup":
        x = 0.65 + data / 250       # based on func_data
        y = a + b * p1/10 - c * p2/10
    if method == "DP":
        x = 0.70 + data / 150       # based on func_data
        y = a + b * p1/10 - c * p2/10
        #y = a + b * (np.power(p1/10, d) + np.power(p2/10, d)) + c * p1 * p2
    return x * y * phi_SD


# optimizing data wise boosting (x=a+data/b)
# Sup/DP: 3x change error 1,2
def opt_data():

    # optimal a, b and the corresponding error
    opt = [0, 0, 1]

    for a in np.arange(0.5, 0.9, 0.01):
        for b in np.delete(np.arange(-1000, 1000, 50), 20):

            # dataset (1...5) and player (1,2) wise error corresponding to actual a,b
            temp = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
            data = [10, 15, 20, 25, 30]

            # dataset wise optimization
            for i in np.arange(np.size(data)):

                # read execution data
                get_result(data[i])

                # privacy parameter wise cummulated error
                error1 = 0
                error2 = 0

                # accummulating error
                for p1 in np.arange(np.size(privparam)):
                    for p2 in np.arange(np.size(privparam)):
                        error1 = error1 + np.square(avgR1DP[p1, p2] -
                                                    func_data(data[i], "DP", avgA1DP[p1, p2], a, b))
                        error2 = error2 + np.square(avgR2DP[p1, p2] -
                                                    func_data(data[i], "DP", avgA2DP[p1, p2], a, b))

                # setting dataset and player wise errors
                temp[i][0] = np.sqrt(error1)/np.size(privparam)
                temp[i][1] = np.sqrt(error2)/np.size(privparam)

            # update optimal value
            if np.mean(temp) < opt[2]:
                opt[0] = a
                opt[1] = b
                opt[2] = np.mean(temp)
    return opt


# optimizing privacy parameter wise boosting (y=a+b*p1/10-c*p2/10)
# Sup/DP: 1x change in error 1,2
#         5x change in aux(1,2)(A,R)
def opt_priv():

#    get_result('')
#    r_avgR1Sup = avgR1Sup
#    r_avgR1DP  = avgR1DP
#    r_avgR2Sup = avgR2Sup
#    r_avgR2DP  = avgR2DP
#    r_avgA1Sup = avgA1Sup
#    r_avgA1DP  = avgA1DP
#    r_avgA2Sup = avgA2Sup
#    r_avgA2DP  = avgA2DP

    # reading all the execution data
    get_result(10)
    r10_avgR1Sup = avgR1Sup
    r10_avgR1DP  = avgR1DP
    r10_avgR2Sup = avgR2Sup
    r10_avgR2DP  = avgR2DP
    r10_avgA1Sup = avgA1Sup
    r10_avgA1DP  = avgA1DP
    r10_avgA2Sup = avgA2Sup
    r10_avgA2DP  = avgA2DP

    get_result(15)
    r15_avgR1Sup = avgR1Sup
    r15_avgR1DP  = avgR1DP
    r15_avgR2Sup = avgR2Sup
    r15_avgR2DP  = avgR2DP
    r15_avgA1Sup = avgA1Sup
    r15_avgA1DP  = avgA1DP
    r15_avgA2Sup = avgA2Sup
    r15_avgA2DP  = avgA2DP

    get_result(20)
    r20_avgR1Sup = avgR1Sup
    r20_avgR1DP  = avgR1DP
    r20_avgR2Sup = avgR2Sup
    r20_avgR2DP  = avgR2DP
    r20_avgA1Sup = avgA1Sup
    r20_avgA1DP  = avgA1DP
    r20_avgA2Sup = avgA2Sup
    r20_avgA2DP  = avgA2DP

    get_result(25)
    r25_avgR1Sup = avgR1Sup
    r25_avgR1DP  = avgR1DP
    r25_avgR2Sup = avgR2Sup
    r25_avgR2DP  = avgR2DP
    r25_avgA1Sup = avgA1Sup
    r25_avgA1DP  = avgA1DP
    r25_avgA2Sup = avgA2Sup
    r25_avgA2DP  = avgA2DP

    get_result(30)
    r30_avgR1Sup = avgR1Sup
    r30_avgR1DP  = avgR1DP
    r30_avgR2Sup = avgR2Sup
    r30_avgR2DP  = avgR2DP
    r30_avgA1Sup = avgA1Sup
    r30_avgA1DP  = avgA1DP
    r30_avgA2Sup = avgA2Sup
    r30_avgA2DP  = avgA2DP

    # optimal a, b, c and the corresponding error
    opt = [0, 0, 0, 0, 1]

    for a in np.arange(0.7, 1.1, 0.01):
        for b in np.arange(0.3, 0.7, 0.01):
            for c in np.arange(-0.4, 0.4, 0.01):
                for d in [2]:

                    # privacy parameter ([0,0.1,0.2,0.4,0,5]x[0,0.1,0.2,0.4,0,5]) wise error
                    temp1 = np.zeros(shape=(6, 6))
                    temp2 = np.zeros(shape=(6, 6))

                    # privparam  wise optimization
                    for p1 in np.arange(0, 6):
                        for p2 in np.arange(0, 6):

#                           temp1[p1][p2] = np.sqrt((np.square(r_avgR1Sup[p1, p2] - func_priv(
#                                                        p1, p2, 20, "Sup", r_avgA1Sup[p1, p2], a, b, c)) +
#                                                    np.square(r_avgR2Sup[p1, p2] - func_priv(
#                                                        p1, p2, 20, "Sup", r_avgA2Sup[p1, p2], a, b, c)))/2)

                            # dataset based structure changed to privparam based structure
                            aux1A = [r10_avgA1Sup[p1, p2], r15_avgA1Sup[p1, p2], r20_avgA1Sup[p1, p2],
                                     r25_avgA1Sup[p1, p2], r30_avgA1Sup[p1, p2]]
                            aux1R = [r10_avgR1Sup[p1, p2], r15_avgR1Sup[p1, p2], r20_avgR1Sup[p1, p2],
                                     r25_avgR1Sup[p1, p2], r30_avgR1Sup[p1, p2]]
                            aux2A = [r10_avgA2Sup[p1, p2], r15_avgA2Sup[p1, p2], r20_avgA2Sup[p1, p2],
                                     r25_avgA2Sup[p1, p2], r30_avgA2Sup[p1, p2]]
                            aux2R = [r10_avgR2Sup[p1, p2], r15_avgR2Sup[p1, p2], r20_avgR2Sup[p1, p2],
                                     r25_avgR2Sup[p1, p2], r30_avgR2Sup[p1, p2]]

                            # dataset wise cummulated error
                            data = [10, 15, 20, 25, 30]
                            error1 = 0
                            error2 = 0

                            # accummulating error
                            for i in [0, 1, 2, 3, 4]:
                                error1 = error1 + \
                                         np.square(aux1R[i] - func_priv(p1, p2, data[i], "Sup", aux1A[i], a, b, c, d))
                                error2 = error2 + \
                                         np.square(aux2R[i] - func_priv(p1, p2, data[i], "Sup", aux2A[i], a, b, c, d))

                            # setting privparam wise errors
                            temp1[p1][p2] = np.sqrt(error1/5)
                            temp2[p1][p2] = np.sqrt(error2/5)

                    # update optimal value
                    if np.mean([np.mean(temp1), np.mean(temp2)]) < opt[4]:
                        opt[0] = a
                        opt[1] = b
                        opt[2] = c
                        opt[3] = d
                        opt[4] = np.mean([np.mean(temp1), np.mean(temp2)])
    return opt


# percentage wise improvement of boosting and combination
# NF: get_result = source
# ML20: get_result = ''
def combine(source):
    get_result(source)
    # boosted values
    aux1S = np.zeros(shape=(6, 6))
    aux2S = np.zeros(shape=(6, 6))
    aux1D = np.zeros(shape=(6, 6))
    aux2D = np.zeros(shape=(6, 6))
    for p1 in np.arange(0, 6):
        for p2 in np.arange(0, 6):
            aux1S[p1][p2] = funcLIN(p1, p2, source, "Sup", avgA1Sup[p1, p2])
            aux2S[p1][p2] = funcLIN(p1, p2, source, "Sup", avgA2Sup[p1, p2])
            aux1D[p1][p2] = funcLIN(p1, p2, source, "DP", avgA1DP[p1, p2])
            aux2D[p1][p2] = funcLIN(p1, p2, source, "DP", avgA2DP[p1, p2])

    # error with boosting
    errO_1S = np.sqrt(np.sum(np.square(avgR1Sup - aux1S)))/6
    errO_2S = np.sqrt(np.sum(np.square(avgR2Sup - aux2S)))/6
    errO_1D = np.sqrt(np.sum(np.square(avgR1DP  - aux1D)))/6
    errO_2D = np.sqrt(np.sum(np.square(avgR2DP  - aux2D)))/6

    # error with combination with boosting
    errC_1S = np.sqrt(np.sum(np.square(avgR1Sup - np.mean([aux1S, aux2S], axis=0))))/6
    errC_2S = np.sqrt(np.sum(np.square(avgR2Sup - np.mean([aux1S, aux2S], axis=0))))/6
    errC_1D = np.sqrt(np.sum(np.square(avgR1DP  - np.mean([aux1D, aux2D], axis=0))))/6
    errC_2D = np.sqrt(np.sum(np.square(avgR2DP  - np.mean([aux1D, aux2D], axis=0))))/6

    # error without boosting
    base_errO_1S = np.sqrt(np.sum(np.square(avgR1Sup - avgA1Sup)))/6
    base_errO_2S = np.sqrt(np.sum(np.square(avgR2Sup - avgA2Sup)))/6
    base_errO_1D = np.sqrt(np.sum(np.square(avgR1DP  - avgA1DP)))/6
    base_errO_2D = np.sqrt(np.sum(np.square(avgR2DP  - avgA2DP)))/6

    # error eith combination without boosting
    base_errC_1S = np.sqrt(np.sum(np.square(avgR1Sup - np.mean([avgA1Sup, avgA2Sup], axis=0))))/6
    base_errC_2S = np.sqrt(np.sum(np.square(avgR2Sup - np.mean([avgA1Sup, avgA2Sup], axis=0))))/6
    base_errC_1D = np.sqrt(np.sum(np.square(avgR1DP  - np.mean([avgA1DP, avgA2DP], axis=0))))/6
    base_errC_2D = np.sqrt(np.sum(np.square(avgR2DP  - np.mean([avgA1DP, avgA2DP], axis=0))))/6

    # improvement percentage of combination without boosting
    print([np.round(100 * (base_errO_1S - base_errC_1S) / base_errO_1S),
           np.round(100 * (base_errO_1D - base_errC_1D) / base_errO_1D),
           np.round(100 * (base_errO_2S - base_errC_2S) / base_errO_2S),
           np.round(100 * (base_errO_2D - base_errC_2D) / base_errO_2D)])

    # improvement percentage of boosting
    print([np.round(100 * (base_errO_1S - errO_1S) / base_errO_1S),
           np.round(100 * (base_errO_1D - errO_1D) / base_errO_1D),
           np.round(100 * (base_errO_2S - errO_2S) / base_errO_2S),
           np.round(100 * (base_errO_2D - errO_2D) / base_errO_2D)])

    # improvement percentage of combination with boosting
    print([np.round(100 * (base_errO_1S - errC_1S) / base_errO_1S),
           np.round(100 * (base_errO_1D - errC_1D) / base_errO_1D),
           np.round(100 * (base_errO_2S - errC_2S) / base_errO_2S),
           np.round(100 * (base_errO_2D - errC_2D) / base_errO_2D)])


# privacy parameter wise errors (accumulated for dataset)
def err_priv(data, player, method):

    # reading execution data
    get_result(data)

    # player and method wise separation
    if player == 1 and method == "Sup":
        auxA = avgA1Sup
        auxR = avgR1Sup
    if player == 1 and method == "DP":
        auxA = avgA1DP
        auxR = avgR1DP
    if player == 2 and method == "Sup":
        auxA = avgA2Sup
        auxR = avgR2Sup
    if player == 2 and method == "DP":
        auxA = avgA2DP
        auxR = avgR2DP

    # accumulated error acording to privparam
    error = 0
    for p1 in np.arange(np.size(privparam)):
        for p2 in np.arange(np.size(privparam)):
            error = error + np.square(auxR[p1, p2] - funcLIN(p1, p2, data, method, auxA[p1, p2]))

    return np.sqrt(error)/np.size(privparam)


# dataset and player wise errors (accumulated for privparam)
def err_data(p1, p2, player, method):

    get_result(10)
    r10_avgR1Sup = avgR1Sup
    r10_avgR1DP  = avgR1DP
    r10_avgR2Sup = avgR2Sup
    r10_avgR2DP  = avgR2DP
    r10_avgA1Sup = avgA1Sup
    r10_avgA1DP  = avgA1DP
    r10_avgA2Sup = avgA2Sup
    r10_avgA2DP  = avgA2DP

    get_result(15)
    r15_avgR1Sup = avgR1Sup
    r15_avgR1DP  = avgR1DP
    r15_avgR2Sup = avgR2Sup
    r15_avgR2DP  = avgR2DP
    r15_avgA1Sup = avgA1Sup
    r15_avgA1DP  = avgA1DP
    r15_avgA2Sup = avgA2Sup
    r15_avgA2DP  = avgA2DP

    get_result(20)
    r20_avgR1Sup = avgR1Sup
    r20_avgR1DP  = avgR1DP
    r20_avgR2Sup = avgR2Sup
    r20_avgR2DP  = avgR2DP
    r20_avgA1Sup = avgA1Sup
    r20_avgA1DP  = avgA1DP
    r20_avgA2Sup = avgA2Sup
    r20_avgA2DP  = avgA2DP

    get_result(25)
    r25_avgR1Sup = avgR1Sup
    r25_avgR1DP  = avgR1DP
    r25_avgR2Sup = avgR2Sup
    r25_avgR2DP  = avgR2DP
    r25_avgA1Sup = avgA1Sup
    r25_avgA1DP  = avgA1DP
    r25_avgA2Sup = avgA2Sup
    r25_avgA2DP  = avgA2DP

    get_result(30)
    r30_avgR1Sup = avgR1Sup
    r30_avgR1DP  = avgR1DP
    r30_avgR2Sup = avgR2Sup
    r30_avgR2DP  = avgR2DP
    r30_avgA1Sup = avgA1Sup
    r30_avgA1DP  = avgA1DP
    r30_avgA2Sup = avgA2Sup
    r30_avgA2DP  = avgA2DP

    # dataset based structure changed to player and method based structure
    if player == 1 and method == "Sup":
        auxA = [r10_avgA1Sup[p1, p2], r15_avgA1Sup[p1, p2], r20_avgA1Sup[p1, p2],
                r25_avgA1Sup[p1, p2], r30_avgA1Sup[p1, p2]]
        auxR = [r10_avgR1Sup[p1, p2], r15_avgR1Sup[p1, p2], [p1, p2],
                r25_avgR1Sup[p1, p2], r30_avgR1Sup[p1, p2]]
    if player == 1 and method == "DP":
        auxA = [r10_avgA1DP[p1, p2], r15_avgA1DP[p1, p2], r20_avgA1DP[p1, p2],
                r25_avgA1DP[p1, p2], r30_avgA1DP[p1, p2]]
        auxR = [r10_avgR1DP[p1, p2], r15_avgR1DP[p1, p2], r20_avgR1DP[p1, p2],
                r25_avgR1DP[p1, p2], r30_avgR1DP[p1, p2]]
    if player == 2 and method == "Sup":
        auxA = [r10_avgA2Sup[p1, p2], r15_avgA2Sup[p1, p2], r20_avgA2Sup[p1, p2],
                r25_avgA2Sup[p1, p2], r30_avgA2Sup[p1, p2]]
        auxR = [r10_avgR2Sup[p1, p2], r15_avgR2Sup[p1, p2], r20_avgR2Sup[p1, p2],
                r25_avgR2Sup[p1, p2], r30_avgR2Sup[p1, p2]]
    if player == 2 and method == "DP":
        auxA = [r10_avgA2DP[p1, p2], r15_avgA2DP[p1, p2], r20_avgA2DP[p1, p2],
                r25_avgA2DP[p1, p2], r30_avgA2DP[p1, p2]]
        auxR = [r10_avgR2DP[p1, p2], r15_avgR2DP[p1, p2], r20_avgR2DP[p1, p2],
                r25_avgR2DP[p1, p2], r30_avgR2DP[p1, p2]]

    # dataset wise accumulated error
    data = [10, 15, 20, 25, 30]
    error = 0

    # accumulating error
    for i in [0, 1, 2, 3, 4]:
        error = error + np.square(auxR[i] - funcLIN(p1, p2, data[i], method, auxA[i]))

    return np.sqrt(error/5)


#formating the data for Mathematica
def converter(data):
    temp = ''
    for i in [0, 1, 2, 3, 4, 5]:
        for j in [0, 1, 2, 3, 4, 5]:
            temp = temp + '{{' + str(i/10) + ',' + str(j/10) + '},' + str(data[i, j]) + '},'
        temp = temp + '\n'
    return print(temp)


# reading the matrix by the diagonals
def diagonalread(matrix):
    direction = True
    k = 0
    i = 0
    j = 0
    res = []
    while k < matrix.size:
        if direction:
            while i >= 0 and j < np.sqrt(matrix.size):
                res.append([i, j])
                j = j + 1
                i = i - 1
                k = k + 1
            if i < 0 and j <= np.sqrt(matrix.size) - 1:
                i = 0
            if j == np.sqrt(matrix.size):
                i = i + 2
                j = j - 1
        else:
            while j >= 0 and i < np.sqrt(matrix.size):
                res.append([i, j])
                k = k + 1
                i = i + 1
                j = j - 1
            if j < 0 and i <= np.round(np.sqrt(matrix.size) - 1):
                j = 0
            if i == np.sqrt(matrix.size):
                j = j + 2
                i = i - 1
        direction = not direction
    return res


# monotonizing a set of 3
def locmon(x, y, z):
    temp = [0, 0, 100]
    if x < y < z:
        a = np.maximum(np.abs(y-x), np.abs(y-z))/20
        return [y + a, y, y - a]
    if x < y > z:
        for a in np.arange((y-z)/20, (y-z)/2, (y-z)/20):
            if np.square(z+2*a-x)+np.square(z+a-y) < temp[2]:
                temp[0] = z+2*a
                temp[1] = z+a
                temp[2] = np.square(z+2*a-x)+np.square(z+a-y)
        return [temp[0], temp[1], z]
    if x > y < z:
        for a in np.arange((x-y)/20, (x-y)/2, (x-y)/20):
            if np.square(x-2*a-z)+np.square(x-a-y) < temp[2]:
                temp[0] = x-2*a
                temp[1] = x-a
                temp[2] = np.square(x-2*a-x)+np.square(x-a-y)
        return [x, temp[1], temp[0]]
    if x > y > z:
        return [x, y, z]


# monotonizing a matrix
def monotonizer(data):
    out = [data-1, data+1]
    o = np.array(data)
    count = 0
    while not np.array_equal(out[0], out[1]):
        for x in diagonalread(o):
            if x[0]+2 <= np.sqrt(o.size)-1:
                temp = locmon(o[x[0], x[1]], o[x[0]+1, x[1]], o[x[0]+2, x[1]])
                o[x[0], x[1]] = temp[0]
                o[x[0]+1, x[1]] = temp[1]
                o[x[0]+2, x[1]] = temp[2]
            if x[1]+2 <= np.sqrt(o.size)-1:
                temp = locmon(o[x[0], x[1]], o[x[0], x[1]+1], o[x[0], x[1]+2])
                o[x[0], x[1]] = temp[0]
                o[x[0], x[1]+1] = temp[1]
                o[x[0], x[1]+2] = temp[2]
        if np.mod(count, 2) == 0:
            out[0] = o
        else:
            out[1] = o
        count = count + 1
    return o


#combine(10)
#combine(15)
#combine(20)
#combine(25)
#combine(30)

#print(err_priv(10, 1, "Sup"), err_priv(15, 1, "Sup"), err_priv(20, 1, "Sup"),
#      err_priv(25, 1, "Sup"), err_priv(30, 1, "Sup"), '\n',
#      err_priv(10, 2, "Sup"), err_priv(15, 2, "Sup"), err_priv(20, 2, "Sup"),
#      err_priv(25, 2, "Sup"), err_priv(30, 2, "Sup"), '\n',
#      err_priv(10, 1, "DP"), err_priv(15, 1, "DP"), err_priv(20, 1, "DP"),
#      err_priv(25, 1, "DP"), err_priv(30, 1, "DP"), '\n',
#      err_priv(10, 2, "DP"), err_priv(15, 2, "DP"), err_priv(20, 2, "DP"),
#      err_priv(25, 2, "DP"), err_priv(30, 2, "DP"), '\n\n')

#for player in [1, 2]:
#    for method in ["Sup", "DP"]:
#        print(
#            [err_data(0, 0, player, method), err_data(0, 1, player, method), err_data(0, 2, player, method),
#             err_data(0, 3, player, method), err_data(0, 4, player, method), err_data(0, 4, player, method)], '\n',
#            [err_data(1, 0, player, method), err_data(1, 1, player, method), err_data(1, 2, player, method),
#             err_data(1, 3, player, method), err_data(1, 4, player, method), err_data(1, 4, player, method)], '\n',
#            [err_data(2, 0, player, method), err_data(2, 1, player, method), err_data(2, 2, player, method),
#             err_data(2, 3, player, method), err_data(2, 4, player, method), err_data(2, 4, player, method)], '\n',
#            [err_data(3, 0, player, method), err_data(3, 1, player, method), err_data(3, 2, player, method),
#             err_data(3, 3, player, method), err_data(3, 4, player, method), err_data(3, 4, player, method)], '\n',
#            [err_data(4, 0, player, method), err_data(4, 1, player, method), err_data(4, 2, player, method),
#             err_data(4, 3, player, method), err_data(4, 4, player, method), err_data(4, 4, player, method)], '\n',
#            [err_data(5, 0, player, method), err_data(5, 1, player, method), err_data(5, 2, player, method),
#             err_data(5, 3, player, method), err_data(5, 4, player, method), err_data(5, 5, player, method)], '\n\n')


R1Sup  = np.zeros(shape=(2, 6, 6))
A1Sup  = np.zeros(shape=(2, 6, 6))
R2Sup  = np.zeros(shape=(2, 6, 6))
A2Sup  = np.zeros(shape=(2, 6, 6))
R1DP  = np.zeros(shape=(2, 6, 6))
A1DP  = np.zeros(shape=(2, 6, 6))
R2DP  = np.zeros(shape=(2, 6, 6))
A2DP  = np.zeros(shape=(2, 6, 6))

monR1Sup  = np.zeros(shape=(2, 6, 6))
monR2Sup  = np.zeros(shape=(2, 6, 6))
monR1DP  = np.zeros(shape=(2, 6, 6))
monR2DP  = np.zeros(shape=(2, 6, 6))

aux1SL = np.zeros(shape=(2, 6, 6))
aux2SL = np.zeros(shape=(2, 6, 6))
aux1DL = np.zeros(shape=(2, 6, 6))
aux2DL = np.zeros(shape=(2, 6, 6))
aux1DQ = np.zeros(shape=(2, 6, 6))
aux2DQ = np.zeros(shape=(2, 6, 6))

source = [10, 20]

for d in np.arange(2):
    get_result(source[d])

    R1Sup[d] = np.mean(real1Sup, axis=0)
    R1DP[d]  = np.mean(real1DP,  axis=0)
    R2Sup[d] = np.mean(real2Sup, axis=0)
    R2DP[d]  = np.mean(real2DP,  axis=0)
    A1Sup[d] = np.mean(approx1Sup, axis=0)
    A1DP[d]  = np.mean(approx1DP,  axis=0)
    A2Sup[d] = np.mean(approx2Sup, axis=0)
    A2DP[d]  = np.mean(approx2DP,  axis=0)

    monR1Sup[d] = monotonizer(R1Sup[d])
    monR2Sup[d] = monotonizer(R2Sup[d])
    monR1DP[d]  = monotonizer(A1Sup[d])
    monR2DP[d]  = monotonizer(A2Sup[d])

    for p1 in np.arange(0, 6):
        for p2 in np.arange(0, 6):
            aux1SL[d, p1, p2] = funcLIN(p1, p2, source[d], "Sup", A1Sup[d, p1, p2])
            aux2SL[d, p1, p2] = funcLIN(p1, p2, source[d], "Sup", A2Sup[d, p1, p2])
            aux1DL[d, p1, p2] = funcLIN(p1, p2, source[d], "DP", A1DP[d, p1, p2])
            aux2DL[d, p1, p2] = funcLIN(p1, p2, source[d], "DP", A2DP[d, p1, p2])
            aux1DQ[d, p1, p2] = funcQUAD(p1, p2, source[d], "DP", A1DP[d, p1, p2])
            aux2DQ[d, p1, p2] = funcQUAD(p1, p2, source[d], "DP", A2DP[d, p1, p2])

    #aux1SL[d] = monotonizer(aux1SL[d])
    #aux2SL[d] = monotonizer(aux2SL[d])
    #aux1DL[d] = monotonizer(aux1DL[d])
    #aux2DL[d] = monotonizer(aux2DL[d])
    #aux1DQ[d] = monotonizer(aux1DQ[d])
    #aux2DQ[d] = monotonizer(aux2DQ[d])

#plt.imshow(np.mean([np.mean(R1Sup-A1Sup,axis=0),np.mean(R2Sup-A2Sup,axis=0)],axis=0),cmap='RdBu',origin='lower')
#plt.imshow(np.mean([np.mean(R1DP-A1DP,axis=0),np.mean(R2DP-A2DP,axis=0)],axis=0),cmap='RdBu',origin='lower')
#np.mean([np.mean(np.mean(R1Sup-A1Sup,axis=1),axis=1),np.mean(np.mean(R2Sup-A2Sup,axis=1),axis=1)],axis=0)
#np.mean([np.mean(np.mean(R1DP-A1DP,axis=1),axis=1),np.mean(np.mean(R2DP-A2DP,axis=1),axis=1)],axis=0)
