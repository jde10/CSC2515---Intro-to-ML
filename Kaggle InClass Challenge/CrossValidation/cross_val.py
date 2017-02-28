import numpy as np
from sklearn.model_selection import KFold

data = np.load('FLICKR.npz')
print data.files

inp_train = data['input_train']
tgt_train = data['target_train']
inp_val = data['input_test']
tgt_val = data['target_test']

X = np.array(range(7000))
kf = KFold(n_splits=2)
mini_batch_input_1 = []
mini_batch_input_2 = []
mini_batch_target_1 = []
mini_batch_target_2 = []

print tgt_train

for train, val in kf.split(X):
    print("%s %s" % (train, val))

for i in range (len(train)):
    mini_batch_input_1.append(inp_train[train[i], :, :, :])
    mini_batch_target_1.append(tgt_train[train[i]])
    mini_batch_input_2.append(inp_train[val[i], :, :, :])
    mini_batch_target_2.append(tgt_train[val[i]])

np.savez('batches_k2', batch_1_input = mini_batch_input_1, batch_2_input = mini_batch_input_2, batch_1_target = mini_batch_target_1, batch_2_target = mini_batch_target_2)