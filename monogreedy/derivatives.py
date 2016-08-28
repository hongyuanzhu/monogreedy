import numpy
import copy


class SecondFirstMap:
    def __init__(self):
        self.map2 = []
        self.map1a = []
        self.map1b = []
        self.map1 = []


def second_over_first(mono_phi_score, min_phi, max_phi, step_phi, tal):

    second_first_map = SecondFirstMap()

    def _map_recursion(phi, min_p, max_p, step_p):

        min_phi0 = min_p[0]
        max_phi0 = max_p[0]
        step_phi0 = step_p[0]

        # leaf case
        if len(min_p) == 1:
            phi.append(0)
            for i in range(min_phi0, max_phi0, step_phi0):
                phi[-1] = i

                value_h, alpha = train_predict_regression(mono_phi_score, phi, tal)
                second_order, first_order1, first_order2, first_order = second_first_first(alpha, phi)

                second_first_map.map2.append(second_order)
                second_first_map.map1a.append(first_order1)
                second_first_map.map1b.append(first_order2)
                second_first_map.map1.append(first_order)

            return

        # general case
        min_phi1 = min_p[1:]
        max_phi1 = max_p[1:]
        step_phi1 = step_p[1:]

        phi.append(0)
        for i in range(min_phi0, max_phi0, step_phi0):
            phi[-1] = i
            _map_recursion(copy.deepcopy(phi), min_phi1, max_phi1, step_phi1)

    ph = []
    _map_recursion(ph, min_phi, max_phi, step_phi)

    return second_first_map


def train_predict_regression(mono_phi_score, dim_h, tal):
    dim_w = len(dim_h)

    y_matrix = mono_phi_score[:, -1]
    x_matrix = numpy.zeros((len(mono_phi_score), (1+dim_w)*dim_w/2+dim_w+1)) # 1, second order, first order
    for i in range(len(x_matrix)):
        itm = mono_phi_score[i]
        fea = numpy.array(1.)
        for h in range(dim_w):
            for j in range(h+1):
                fea = numpy.append(fea, itm[h]*itm[j])
        for j in range(dim_w):
            fea = numpy.append(fea, itm[j])
        x_matrix[i,:] = fea

    # delta
    delta = numpy.exp(-numpy.sum((mono_phi_score[:, :dim_w] - dim_h)**2, axis=1)/2./tal/tal)
    w_matrix = numpy.diag(delta)
    alpha = numpy.dot(numpy.dot(numpy.dot(numpy.linalg.pinv(numpy.dot(numpy.dot(numpy.transpose(x_matrix), w_matrix),
                                        x_matrix)), numpy.transpose(x_matrix)), w_matrix), y_matrix)

    # predict
    fea = numpy.array(1.)
    for h in range(dim_w):
        for j in range(h+1):
            fea = numpy.append(fea, dim_h[h]*dim_h[j])
    for j in range(dim_w):
        fea = numpy.append(fea, dim_h[j])
    value_h = numpy.dot(alpha, fea)

    return value_h, alpha


def second_first_first(alpha, dim_h):
    dim_w = len(dim_h)

    beta = alpha[-dim_w:]
    alpha_matrix = numpy.zeros((dim_w, dim_w))
    count = 1
    for h in range(dim_w):
        for j in range(h+1):
            alpha_matrix[h][j] = alpha[count]
            count += 1

    second_order = numpy.zeros(((1+dim_w)*dim_w/2,))
    count = 0
    for h in range(dim_w):
        for j in range(h+1):
            second_order[count] = alpha_matrix[h][j]
            if h == j:
                second_order[count] += alpha_matrix[h][j]
            count += 1

    first_order = numpy.zeros((dim_w,))
    for i in range(dim_w):
        first_order[i] = 2*alpha_matrix[i][i]*dim_h[i] + beta[i]
        if i+1 <= dim_w-1:
            first_order[i] += numpy.dot(alpha_matrix[i+1:, i], dim_h[i+1:])
        if i-1 >= 0:
            first_order[i] += numpy.dot(alpha_matrix[i, :i], dim_h[:i])

    first_order1 = numpy.zeros(((1+dim_w)*dim_w/2,))
    count = 0
    for h in range(dim_w):
        for j in range(h+1):
            first_order1[count] = first_order[h]
            count += 1

    first_order2 = numpy.zeros(((1+dim_w)*dim_w/2,))
    count = 0
    for h in range(dim_w):
        for j in range(h+1):
            first_order2[count] = first_order[j]
            count += 1

    return second_order, first_order1, first_order2, first_order


def tune_tal(mono_phi_score, tal_list):
    errs = []
    tals = []
    for tal in tal_list:
        err = []
        for i in range(len(mono_phi_score)):
            mono_1 = numpy.delete(mono_phi_score, i, axis=0)
            dim_h = mono_phi_score[i][:-1]
            value_h, alpha = train_predict_regression(mono_1, dim_h, tal)
            err.append((value_h - mono_phi_score[i][-1])**2)
        err = numpy.mean(err)

        errs.append(err)
        tals.append(tal)
        print 'regression tal:', tal, 'err', err

    idx = numpy.argmin(errs)

    return tals[idx]
