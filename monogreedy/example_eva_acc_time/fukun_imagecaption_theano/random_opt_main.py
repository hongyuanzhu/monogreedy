from train import *
import pickle
import numpy

score_time_phi = [] # load table, must eva smallest and largest
while True:
    # na, nnh, nh, nw = phi
    phi = []
    for j in range(4): #change 3 last
        dim_fea = numpy.random.randint(100, 1000)
        phi.append(dim_fea)

    # eva phi
    scores, running_time = eva_a_phi(phi)
    score_time_phi.append([scores, running_time, phi])

    # save
    output = open('table0.pkl', 'wb')
    pickle.dump(score_time_phi, output)
    output.close()


def collect_table(table_name):

    table_name = 'table0_2.pkl'

    inputf = open(table_name,'rb')
    one_table = pickle.load(inputf)
    inputf.close()

    file = open(table_name[:-3]+'txt', 'w')
    for i in range(len(one_table)):
        score = str(one_table[i][0][4])
        rtime = str(one_table[i][1])
        phi = one_table[i][2]
        phi_str = ''
        for j in range(len(phi)):
            phi_str += str(phi[j])+' '

        file.write(score+' '+rtime+' '+phi_str+'\n')
    file.close()

    return one_table