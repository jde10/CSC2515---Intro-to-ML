import matplotlib.pyplot as plt
import numpy as np

from skimage.feature import hog
from skimage import data, color, exposure

data = np.load('batches_k2.npz')
print data.files

inp_train = data['batch_1_input']
tgt_train = data['batch_1_target']
inp_val = data['batch_2_input']
tgt_val = data['batch_2_target']

print tgt_train

inp_train_gr = []
inp_val_gr = []

fd_train = []
hog_train = []

fd_val = []
hog_val = []


for i in range(inp_train.shape[0]):
    #print tgt_train[i]
    inp_train_gr.append(color.rgb2gray(inp_train[i,:,:,:]))
    inp_val_gr.append(color.rgb2gray(inp_val[i,:,:,:]))
    imaget = inp_train_gr[i]
    fd, hog_image = hog(imaget, orientations=8, pixels_per_cell=(4, 4), cells_per_block = (1,1), visualise = True)

    fd_train.append(fd)
    hog_train.append(hog_image)

    imagev = inp_val_gr[i]
    fdv, hog_imagev = hog(imagev, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1),
                        visualise=True)

    fd_val.append(fdv)
    hog_val.append(hog_imagev)

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    #
    # ax1.axis('off')
    # ax1.imshow(imaget, cmap=plt.cm.gray)
    # ax1.set_title('Input image')
    # ax1.set_adjustable('box-forced')
    #
    # # Rescale histogram for better display
    # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    #
    # ax2.axis('off')
    # ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    # ax2.set_title('Histogram of Oriented Gradients')
    # ax1.set_adjustable('box-forced')
    # plt.show()

np.savez('HOG_features', batch_1_HOG = hog_train, batch_2_HOG = hog_val, batch_1_target = tgt_train, batch_2_target = tgt_val)