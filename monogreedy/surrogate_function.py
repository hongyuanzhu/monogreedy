import numpy


def score_bound(mono_phi_score, dim_h):
    dim_w = len(dim_h)

    n_bound = numpy.array([numpy.max(mono_phi_score[:,-1])])
    # naive bound
    for i in range(len(mono_phi_score)):
        itmA = mono_phi_score[i]
        if sum(itmA[:dim_w] >= dim_h) == dim_w:
            n_bound = numpy.append(n_bound, itmA[-1])
    n_bound = numpy.min(n_bound)

    # naive lower bound
    n_low_bound = numpy.array([numpy.min(mono_phi_score[:,-1])])
    for i in range(len(mono_phi_score)):
        itmA = mono_phi_score[i]
        if sum(itmA[:dim_w] <= dim_h) == dim_w:
            n_low_bound = numpy.append(n_low_bound, itmA[-1])
    n_low_bound = numpy.max(n_low_bound)

    s_bound = numpy.array([n_bound])
    # first bound
    for i in range(len(mono_phi_score)):
        itmA = mono_phi_score[i]
        if not (sum(itmA[:dim_w] >= dim_h) == dim_w and sum(itmA[:dim_w])>sum(dim_h)):
            continue
        for j in range(len(mono_phi_score)):
            itmB = mono_phi_score[j]
            if not sum(itmA[:dim_w] >= itmB[:dim_w]) == dim_w:
                nu = (itmA[:dim_w] - dim_h)*((itmA[:dim_w] - dim_h)>0)
                de = (itmB[:dim_w]-itmA[:dim_w])*((itmB[:dim_w]-itmA[:dim_w])>0)
                alpha = numpy.min(nu/(de+1e-5))
                s_bound = numpy.append(s_bound, itmA[-1]-alpha*(itmB[-1]-itmA[-1]))

    # second bound
    for i in range(len(mono_phi_score)):
        itmA = mono_phi_score[i]
        if sum(itmA[:dim_w] >= dim_h) == dim_w:
            continue
        for j in range(len(mono_phi_score)):
            itmB = mono_phi_score[j]
            if sum(itmA[:dim_w] >= itmB[:dim_w]) == dim_w and sum(itmA[:dim_w])>sum(itmB[:dim_w]):
                nu = (itmA[:dim_w] - itmB[:dim_w])*((itmA[:dim_w] - itmB[:dim_w])>0)
                de = (dim_h-itmA[:dim_w])*((dim_h-itmA[:dim_w])>0)
                alpha = numpy.min(nu/(de+1e-5))
                s_bound = numpy.append(s_bound, itmA[-1]+(itmA[-1]-itmB[-1])/(alpha+1e-5))

    s_bound = numpy.append(s_bound, n_bound)
    # delete wrong bound
    for i in range(len(s_bound)-1,-1,-1):
        if s_bound[i] < n_low_bound or s_bound[i] > n_bound:
            s_bound = numpy.delete(s_bound, i)
    s_bound = numpy.min(s_bound) #mean?

    return s_bound, n_bound # uppser


def time_bound(mono_phi_time, dim_h):
    dim_w = len(dim_h)

    n_bound = numpy.array([numpy.max(mono_phi_time[:,-1])])
    # naive bound
    for i in range(len(mono_phi_time)):
        itmA = mono_phi_time[i]
        if sum(itmA[:dim_w] >= dim_h) == dim_w:
            n_bound = numpy.append(n_bound, itmA[-1])
    n_bound = numpy.min(n_bound)

    # naive lower bound
    n_low_bound = numpy.array([numpy.min(mono_phi_time[:,-1])])
    for i in range(len(mono_phi_time)):
        itmA = mono_phi_time[i]
        if sum(itmA[:dim_w] <= dim_h) == dim_w:
            n_low_bound = numpy.append(n_low_bound, itmA[-1])
    n_low_bound = numpy.max(n_low_bound)

    s_bound = numpy.array([])
    # first bound
    for i in range(len(mono_phi_time)):
        itmA = mono_phi_time[i]
        if not (sum(itmA[:dim_w] <= dim_h) == dim_w and sum(itmA[:dim_w])<sum(dim_h)):
            continue
        for j in range(len(mono_phi_time)):
            itmB = mono_phi_time[j]
            if not sum(itmA[:dim_w] <= itmB[:dim_w]) == dim_w:
                nu = (dim_h - itmA[:dim_w])*((dim_h - itmA[:dim_w])>0)
                de = (itmA[:dim_w]-itmB[:dim_w])*((itmA[:dim_w]-itmB[:dim_w])>0)
                alpha = numpy.min(nu/(de+1e-5))
                s_bound = numpy.append(s_bound, itmA[-1]+alpha*(itmA[-1]-itmB[-1]))

    # second bound
    for i in range(len(mono_phi_time)):
        itmA = mono_phi_time[i]
        if sum(itmA[:dim_w] <= dim_h) == dim_w:
            continue
        for j in range(len(mono_phi_time)):
            itmB = mono_phi_time[j]
            if sum(itmA[:dim_w] <= itmB[:dim_w]) == dim_w and sum(itmA[:dim_w])<sum(itmB[:dim_w]):
                nu = (itmB[:dim_w] - itmA[:dim_w])*((itmB[:dim_w] - itmA[:dim_w])>0)
                de = (itmA[:dim_w]-dim_h)*((itmA[:dim_w]-dim_h)>0)
                alpha = numpy.min(nu/(de+1e-5))
                s_bound = numpy.append(s_bound, itmA[-1]-(itmB[-1]-itmA[-1])/(alpha+1e-5))

    s_bound = numpy.append(s_bound, n_low_bound)
    # delete wrong bound
    for i in range(len(s_bound)-1,-1,-1):
        if s_bound[i] < n_low_bound or s_bound[i] > n_bound:
            s_bound = numpy.delete(s_bound, i)
    s_bound = numpy.max(s_bound)

    return s_bound, n_low_bound #lower
