import numpy as np
from eva_time import *

def eva_a_acc(phi):

    # this is toy data
    # obey acc assumption

    beta = np.array([2+i*0.12 for i in range(len(phi))])/140.
    acc = 100. - 1.*np.exp(4.5 - np.dot(phi, beta))

    return acc

    # you should write your own acc evaluation function
    # an example outline:

    # load data
    # train model
    # traverse data and compute average accuracy
    # return acc


def eva_acc_time(phi):
    acc = eva_a_acc(phi)
    tim = eva_a_time(phi)

    return [acc, tim]


# example eva_a_acc
