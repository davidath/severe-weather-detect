import numpy as np
import sys
import os
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import scale

arr = []
for i in sorted(os.listdir(sys.argv[1])):
    print i
    g = np.load(sys.argv[1] + '/' + i)[sys.argv[2]]
    for i in g:
        curr = i.reshape(1, g.shape[1], g.shape[2]*g.shape[3])
        curr_res = []
        for j in range(g.shape[1]):
            curr_res.append(minmax_scale(curr[0, j, :], feature_range=(0,1)))
            #  curr_res.append(curr[0, j, :])
        curr_res = np.array(curr_res).reshape(g.shape[1] * g.shape[2] * g.shape[3])
        arr.append(curr_res)
arr = np.array(arr)
print np.max(arr)
print np.min(arr)
print np.max(arr[0, :4096])
print np.min(arr[0, :4096])
print np.max(arr[0, 4096*1:4096*2])
print np.min(arr[0, 4096*1:4096*2])
print np.max(arr[0, 4096*2:4096*3])
print np.min(arr[0, 4096*2:4096*3])
# np.savez_compressed('ghtall.npz', ght=arr)
np.savez_compressed('GHT01.npz', train_data=arr, test_data=np.zeros(shape=(1,1)), train_labels=np.zeros(shape=(1,1)), test_labels=np.zeros(shape=(1,1)))
