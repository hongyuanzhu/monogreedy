
import caffe
import numpy
#from pylab import *
import time
import string

# preference
f = open('preference.txt', 'r')
GPUId = string.atoi( f.readline() )
rootfolder = f.readline()
f.close()

asPath = '/home/jinjunqi/wtxP/' + rootfolder + '/autosave/'

caffe.set_mode_cpu()

# We create a solver that fine-tunes from a previously trained network.
solver = caffe.SGDSolver('/home/jinjunqi/wtxP/' + rootfolder + '/sg_solver.prototxt') 
solver.test_nets[0].copy_from(asPath + 'cpu.caffemodel')

#%% time
def tic():
    globals()['tt'] = time.clock()

def toc():
    return time.clock()-globals()['tt']
    
    
niter = 0    
tic()

while 1:
    for i in range(10):
        solver.test_nets[0].forward()
    niter += 1
    if toc()>60:
        break
    
dtime = toc()   

tpi = numpy.array(dtime/niter/10/100 * 1e5) # 1e5 seconds
numpy.save(asPath + 'tpi.npy', tpi)