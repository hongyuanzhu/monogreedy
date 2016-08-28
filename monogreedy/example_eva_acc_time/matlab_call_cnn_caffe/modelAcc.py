#%%
import caffe
#from pylab import *
import os
import numpy
import string

#%%
# preference
f = open('preference.txt', 'r')
GPUId = string.atoi( f.readline() )
rootfolder = f.readline()
f.close()

asPath = '/home/jinjunqi/wtxP/' + rootfolder + '/autosave/'
os.system('rm ' + asPath + '*')

#%%
caffe.set_device(GPUId)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('/home/jinjunqi/wtxP/' + rootfolder + '/sg_solver.prototxt')    

#%%
niter = 0
test_interval = 500
test_acc = [0]
maxIdx = 0
lenTest = 0

while 1:
    solver.step(1)  # SGD by Caffe
    
    if niter % test_interval == 0:
        print 'Iteration', niter, 'testing..', test_acc[-1], maxIdx, lenTest
        
        correct = 0
        for test_it in range(100):
            solver.test_nets[0].forward()
            correct += sum(solver.test_nets[0].blobs['ip2'].data.argmax(1)
                           == solver.test_nets[0].blobs['label'].data)
        test_acc.append(correct / 1e4)
        
        maxIdx = numpy.array(test_acc).argmax() # start from 0
        lenTest = len(test_acc)
        if lenTest-maxIdx>50:
            break
    
    niter += 1
    
#%%    
fileName = 'sg_iter_' + str((niter // 5000)*5000) + '.caffemodel'
os.system('mv ' + asPath + fileName + ' ' + asPath + 'cpu.caffemodel') # for cpu time

#%%
acc = numpy.array(test_acc[maxIdx]*1e2)
numpy.save(asPath+'acc.npy', acc)
