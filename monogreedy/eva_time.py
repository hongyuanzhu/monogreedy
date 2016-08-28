import numpy as np


def eva_a_time(phi):

    # this is toy data
    # obey time assumption
    fea = [1]
    for i in range(len(phi)-1):
        fea.append(phi[i])
        fea.append(phi[i]*phi[i+1])

    fea.append(phi[-1])

    beta = [1+i*0.1 for i in range(len(fea))]

    return np.dot(fea, beta)

    # you should write your own time evaluation function
    # an example outline:

    # load data
    # load model
    # start timer
    # traverse data several times
    # stop timer
    # return time


# example eva_a_time

