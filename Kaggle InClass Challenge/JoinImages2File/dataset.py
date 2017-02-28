import numpy as np
from matplotlib import pyplot as plt
import csv
import os
from scipy import ndimage, misc
import re

data = np.load('toronto_face.npz')

print data.files

tt = data['target_train']
it =  data['inputs_train']
im = it#.reshape(it.shape[0], 48, 48)

it2 = im#.reshape(im.shape[0], 48*48)

im2 = it2.reshape(it2.shape[0], 48, 48)
print it2.shape

# print it.shape
# plt.figure(1)
# plt.imshow(im[0,:,:], cmap=plt.cm.gray)
# plt.savefig('plot')
#
# plt.figure(2)
# plt.imshow(im2[0,:,:], cmap=plt.cm.gray)
# plt.savefig('plot2')
# print im.shape

with open ('train.csv', mode='r') as file:
    reader = csv.reader(file)
    mydict = {rows[0]:rows[1] for rows in reader}
    print(mydict)

# mini_input_train = []
# mini_target_train = []
# for root, dirnames, filenames, in os.walk("./train"):
#     for filename in filenames:
#         if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
#             filepath = os.path.join(root, filename)
#             image = ndimage.imread(filepath, mode="RGB")
#             #image_resized = misc.imresize(image, (64, 64))
#             mini_input_train.append(image)#_resized)
#             key= """+ +"""
#             mini_target_train.append(mydict.get(str(int(filename[1:5]))))


mini_input_val = []
mini_target_val = []
for root, dirnames, filenames, in os.walk("./val"):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            filepath = os.path.join(root, filename)
            image = ndimage.imread(filepath, mode="RGB")
            #image_resized = misc.imresize(image, (64, 64))
            mini_input_val.append(image)#_resized)
            mini_target_val.append(str(int(filename[1:5])))
            #mini_target_val.append(mydict.get(str(int(filename[1:5]))))

#mini_input_train = np.array(mini_input_train)
#mini_target_train = np.array(mini_target_train)
mini_input_val = np.array(mini_input_val)
mini_target_val = np.array(mini_target_val)

#print mini_target_train

#print mini_input_train.shape
#print mini_target_train.shape
print mini_input_val.shape
print mini_target_val.shape

#mini_input_train = mini_input_train.reshape(mini_input_train.shape[0], 64*64*3)

#mini_input_val = mini_input_val.reshape(mini_input_val.shape[0], 64*64*3)

np.savez('FLICKR_test', input_test = mini_input_val, target_test = mini_target_val)
#np.savez('FLICKR', input_train = mini_input_train, input_test = mini_input_val, target_train = mini_target_train, target_test = mini_target_val)