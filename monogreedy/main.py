# run main.py to see the demo results
# change eva_acc.py and eva_time.py to adapt to new applications

from tools import *
from analyzer import *
from opt_over_surrogate import *

# grid seeds
min_phi = [0, 0]
max_phi = [100, 100]
step_phi = [25, 25]
eva_history = Eva_history()

phi = []
eva_recursion(eva_history, phi, min_phi, max_phi, step_phi)

# show acc and time surface
X = np.arange(0, 100, 25)
Y = np.arange(0, 100, 25)
draw2dsurface(X, Y, eva_a_acc)  # w1, w2, acc
draw2dsurface(X, Y, eva_a_time)  # w1, w2, time

# loop
for itr in range(10):
    # make data monotone increasing
    mono_acc_history, mono_time_history = monotone_history(eva_history)

    # function transformation
    num_point_each_dim = 5
    first_order_eps = 1e-5
    time_budget0 = 5000 # suppose 5200 budget
    mono_phi_score, mono_phi_time, time_budget = function_transformation(
        mono_acc_history, mono_time_history, num_point_each_dim, first_order_eps, time_budget0)

    # upper and lower bound, submodular optimization
    num_eva = 3
    eva_candidate = opt_cand(mono_phi_score, mono_phi_time, time_budget, num_eva)

    # new evaluation to history
    update_eva_history(eva_history, eva_candidate)


# acc vs num_eva
best_points = []
for i in range(1,len(eva_history.acc),1):
    points_acc_time = np.concatenate((eva_history.phi[:i],
                                      np.reshape(eva_history.acc[:i],(i,1)),
                                      np.reshape(eva_history.time[:i], (i,1))), 1).tolist()

    for j in range(len(points_acc_time)-1, -1, -1):
        if points_acc_time[j][-1] > time_budget0:
            del points_acc_time[j]

    idx = np.argmax(np.array(points_acc_time)[:, -2])
    best_points.append(points_acc_time[idx])

best_points


# random search, acc vs num_eva
rand_points = []
for i in range(len(eva_history.acc)):
    phi = []
    for j in range(len(min_phi)):
        phi.append(np.random.randint(0, 200))

    [acc, time] = eva_acc_time(phi)
    if time > time_budget0:
        acc = 0

    phi.extend([acc, time])
    rand_points.append(copy.deepcopy(phi))

    idx = np.argmax(np.array(rand_points)[:,-2])
    rand_points[-1] = copy.deepcopy(rand_points[idx])

rand_points

print '30 points, 5000 budget, random search:'
print 'phi:', rand_points[-1][:-2], 'acc:', rand_points[-1][-2], 'time:', rand_points[-1][-1]
print
print '30 points, 5000 budget, submodular search:'
print 'phi:', best_points[-1][:-2], 'acc:', best_points[-1][-2], 'time:', best_points[-1][-1]

