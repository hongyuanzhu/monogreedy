import numpy as np
import copy
from eva_acc import *
from derivatives import *


class Eva_history:
    def __init__(self):
        self.phi = []
        self.time = []
        self.acc = []


def eva_recursion(eva_history, phi, min_phi, max_phi, step_phi):

    min_phi0 = min_phi[0]
    max_phi0 = max_phi[0]
    step_phi0 = step_phi[0]

    # leaf case
    if len(min_phi) == 1:
        phi.append(0)
        for i in range(min_phi0, max_phi0, step_phi0):
            phi[-1] = i

            eva_history.phi.append(copy.deepcopy(phi))
            eva_history.time.append(copy.deepcopy(eva_a_time(phi)))
            eva_history.acc.append(copy.deepcopy(eva_a_acc(phi)))

        return

    # general case
    min_phi1 = min_phi[1:]
    max_phi1 = max_phi[1:]
    step_phi1 = step_phi[1:]

    phi.append(0)
    for i in range(min_phi0, max_phi0, step_phi0):
        phi[-1] = i
        eva_recursion(eva_history, copy.deepcopy(phi), min_phi1, max_phi1, step_phi1)


def monotone_history(eva_history): #(dim_h, score, time)
    phi_score_time = []
    for i in range(len(eva_history.acc)):
        itm = copy.deepcopy(eva_history.phi[i])
        itm.append(eva_history.acc[i])
        itm.append(eva_history.time[i])

        phi_score_time.append(itm)
    phi_score_time = np.array(phi_score_time)


    dim_w = len(phi_score_time[0])-2
    mono_phi_score = np.concatenate((phi_score_time[:, :dim_w],
                                        np.reshape(phi_score_time[:, -2], (len(phi_score_time),1))), axis=1)
    mono_phi_time = np.concatenate((phi_score_time[:, :dim_w],
                                       np.reshape(phi_score_time[:, -1], (len(phi_score_time),1))), axis=1)

    # score, if small phi is better, delete current
    for i in range(len(mono_phi_score)-1, -1, -1):
        current = mono_phi_score[i]
        delete_flag = 0
        for j in range(len(mono_phi_score)):
            itm = mono_phi_score[j]
            if sum(current[:dim_w] >= itm[:dim_w]) == dim_w and itm[-1] >= current[-1] and i != j:
                delete_flag = 1
                break
        if delete_flag == 1:
            mono_phi_score = np.delete(mono_phi_score, i, axis=0)

    # time, if small phi is better, delete current
    for i in range(len(mono_phi_time)-1, -1, -1):
        current = mono_phi_time[i]
        delete_flag = 0
        for j in range(len(mono_phi_time)):
            itm = mono_phi_time[j]
            if sum(current[:dim_w] <= itm[:dim_w]) == dim_w and itm[-1] <= current[-1] and i != j:
                delete_flag = 1
                break
        if delete_flag == 1:
            mono_phi_time = np.delete(mono_phi_time, i, axis=0)

    mono_acc_history = Eva_history()
    for i in range(len(mono_phi_score)):
        mono_acc_history.phi.append(mono_phi_score[i, :-1])
        mono_acc_history.acc.append(mono_phi_score[i, -1])

    mono_time_history = Eva_history()
    for i in range(len(mono_phi_time)):
        mono_time_history.phi.append(mono_phi_time[i, :-1])
        mono_time_history.time.append(mono_phi_time[i, -1])

    return mono_acc_history, mono_time_history


def explode(z):
    return np.exp(z) - 1


def saturate(z):
    return 1 - np.exp(-z)


def function_transformation(mono_acc_history, mono_time_history, num_point_each_dim, first_order_eps, time_budget):
    # acc
    # phi range
    min_phi = np.min(mono_acc_history.phi, 0).astype('int32')
    max_phi = np.max(mono_acc_history.phi, 0).astype('int32')
    step_phi = ((max_phi - min_phi)/num_point_each_dim).astype('int32')

    mono_phi_score = []
    for i in range(len(mono_acc_history.acc)):
        itm = copy.deepcopy(mono_acc_history.phi[i])
        itm = np.append(itm, mono_acc_history.acc[i])

        mono_phi_score.append(itm)
    mono_phi_score = np.array(mono_phi_score)

    # derivatives
    min_tal_scale = np.min((max_phi/10).astype('int32'))
    max_tal_scale = np.max(max_phi.astype('int32'))
    step_tal_scale = ((max_tal_scale - min_tal_scale)/10).astype('int32')

    tal = tune_tal(mono_phi_score, range(min_tal_scale, max_tal_scale, step_tal_scale))
    second_first_map = second_over_first(mono_phi_score, min_phi, max_phi, step_phi, tal)

    # remove no increasing region, overfitting region
    for i in range(len(second_first_map.map1)-1, -1, -1):
        if np.min(second_first_map.map1[i]) < first_order_eps:
            del second_first_map.map1[i]
            del second_first_map.map2[i]
            del second_first_map.map1a[i]
            del second_first_map.map1b[i]

    s_over_f = []
    for i in range(len(second_first_map.map1)):
        s_over_f.extend(second_first_map.map2[i]/second_first_map.map1a[i]/second_first_map.map1b[i])

    # for acc
    mu_acc = np.max(s_over_f)
    if mu_acc<0:
        mu = -mu_acc  # less saturated
        gamma = explode
    else:
        mu = mu_acc  # more saturated
        gamma = saturate

    for i in range(len(mono_phi_score)):
        mono_phi_score[i][-1] = gamma(mono_phi_score[i][-1]*mu)/mu

    # time
    # phi range
    min_phi = np.min(mono_time_history.phi, 0).astype('int32')
    max_phi = np.max(mono_time_history.phi, 0).astype('int32')
    step_phi = ((max_phi - min_phi)/num_point_each_dim).astype('int32')

    mono_phi_time = []
    for i in range(len(mono_time_history.time)):
        itm = copy.deepcopy(mono_time_history.phi[i])
        itm = np.append(itm, mono_time_history.time[i])

        mono_phi_time.append(itm)
    mono_phi_time = np.array(mono_phi_time)

    # derivatives
    min_tal_scale = np.min((max_phi/10).astype('int32'))
    max_tal_scale = np.max(max_phi.astype('int32'))
    step_tal_scale = ((max_tal_scale - min_tal_scale)/10).astype('int32')

    tal = tune_tal(mono_phi_time, range(min_tal_scale, max_tal_scale, step_tal_scale))
    second_first_map = second_over_first(mono_phi_time, min_phi, max_phi, step_phi, tal)

    # remove no increasing region, overfitting region
    for i in range(len(second_first_map.map1)-1, -1, -1):
        if np.min(second_first_map.map1[i]) < first_order_eps:
            second_first_map.map1.remove(i)
            second_first_map.map2.remove(i)
            second_first_map.map1a.remove(i)
            second_first_map.map1b.remove(i)

    s_over_f = []
    for i in range(len(second_first_map.map1)):
        s_over_f.extend(second_first_map.map2[i]/second_first_map.map1a[i]/second_first_map.map1b[i])

    # for time
    mu_time = np.min(s_over_f)
    if mu_time > 0:
        mu = mu_time  # less explode
        gamma = saturate
    else:
        mu = -mu_time  # more explode
        gamma = explode

    for i in range(len(mono_phi_time)):
        mono_phi_time[i][-1] = gamma(mono_phi_time[i][-1]*mu)/mu

    return mono_phi_score, mono_phi_time, gamma(time_budget*mu)/mu


def update_eva_history(eva_history, eva_candidate):

    for i in range(len(eva_candidate)):
        phi = eva_candidate[i]

        continue_flag = 0
        for j in range(len(eva_history.phi)):
            if numpy.sum(numpy.abs(phi - eva_history.phi[j])) < 1e-4:
                continue_flag = 1
                break
        if continue_flag == 1:
            continue

        eva_history.phi.append(phi.tolist())
        eva_history.time.append(eva_a_time(phi))
        eva_history.acc.append(eva_a_acc(phi))

