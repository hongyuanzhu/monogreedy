import numpy
from surrogate_function import *
import copy
from eva_acc import *


def opt_cand(mono_phi_score, mono_phi_time, time_budget, num_eva):

    evas = []
    s_bests = []
    for step in range(1, 10, 2):
        eva_candidate, s_best = opt_candidates(mono_phi_score, mono_phi_time, time_budget, step, num_eva)
        evas.append(eva_candidate)
        s_bests.append(s_best)

    idx = numpy.argmax(s_bests)
    return evas[idx]


def opt_candidates(mono_phi_score, mono_phi_time, time_budget, step, num_eva):

    # opt track
    opt_track = [numpy.array([10, 100])]
    while True:
        cur = opt_track[-1]
        s0, n_bound = score_bound(mono_phi_score, cur)
        t0, n_low_bound = time_bound(mono_phi_time, cur)
        [s0, t0] = eva_acc_time(cur)  # comment

        # increase
        ds = []
        delta_flag = cur*0+1
        for i in range(len(cur)):
            dim_h = numpy.array(copy.deepcopy(cur))
            dim_h[i] = dim_h[i]+step

            si, n_bound = score_bound(mono_phi_score, dim_h)
            ti, n_low_bound = time_bound(mono_phi_time, dim_h)
            [si, ti] = eva_acc_time(dim_h)  # comment

            si = numpy.max([si, s0])
            ti = numpy.max([ti, t0])

            if ti > time_budget:
                delta_flag[i] = 0

            delta = (si - s0) / (ti - t0 + 1e-8)
            ds.append(delta)

        if delta_flag.sum() == 0:
            break

        ds = ds*delta_flag
        idx = numpy.argmax(ds)
        val = ds[idx]
        cand = []
        for i in range(len(ds)):
            if val - ds[i] < 1e-5:
                cand.append(i)

        id = numpy.random.randint(0, len(cand))
        inc_dim = cand[id]

        # save track
        dim_h = numpy.array(copy.deepcopy(cur))
        dim_h[inc_dim] = dim_h[inc_dim]+step
        print 'opt track:', dim_h
        opt_track.append(dim_h)

    # select eva points
    eva_candidate = [opt_track[-1]]

    opt_track = opt_track[:-1]
    for i in range(num_eva-1):
        idx = numpy.random.randint(0, len(opt_track))
        eva_candidate.append(opt_track[idx])

    s_best, n_bound = score_bound(mono_phi_score, eva_candidate[0])
    return eva_candidate, s_best


