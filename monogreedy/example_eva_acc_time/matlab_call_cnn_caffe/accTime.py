#%%
#from pylab import *
import os
import numpy

from archiGen import *


def accAndTime(convoList, fullList, rootfolder):   
    
    #%% cpu.caffemodel and acc.npy
    with open('/home/jinjunqi/wtxP/' + rootfolder + '/sg_train.prototxt', 'w') as f:
        f.write(str(archiEncoder(convoList, fullList,\
        '/home/jinjunqi/caffeLearn/caffe/examples/mnist/mnist_train_lmdb', 64)))
    
    with open('/home/jinjunqi/wtxP/' + rootfolder + '/sg_test.prototxt', 'w') as f:
        f.write(str(archiEncoder(convoList, fullList,\
        '/home/jinjunqi/caffeLearn/caffe/examples/mnist/mnist_test_lmdb', 100)))
        
    os.system('ipython modelAcc.py')   
    
    #%% time.npy
    asPath = '/home/jinjunqi/wtxP/' + rootfolder + '/autosave/' 
    
    os.system('ipython modelTime.py')
    
    #%% return value
    acc = numpy.load(asPath + 'acc.npy')
    tpi = numpy.load(asPath + 'tpi.npy')
    return {'acc': acc, 'tpi': tpi}
