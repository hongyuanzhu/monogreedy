import string
import numpy
from accTime import accAndTime

# archi
f = open('archi.txt', 'r')
convoStages = string.atoi( f.readline() )
fullStages = string.atoi( f.readline() )

convoList = []
for i in range(1,convoStages+1):
    width = string.atoi( f.readline() )
    convoList.append(width)

fullList = []
for i in range(1,fullStages+1):
    width = string.atoi( f.readline() )
    fullList.append(width)   

f.close()

# preference
f = open('preference.txt', 'r')
GPUId = string.atoi( f.readline() )
rootfolder = f.readline()
f.close()

#%%
acctpi = accAndTime(convoList, fullList, rootfolder)

#
f = open('accTime.txt', 'w')
f.write(str(acctpi['acc']) + '\n')
f.write(str(acctpi['tpi']))
f.close()
