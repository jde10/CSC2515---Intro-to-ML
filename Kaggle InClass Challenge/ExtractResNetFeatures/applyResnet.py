from resnet50 import ResNet50
import numpy as np
from keras.preprocessing import image
from imagenet_utils import decode_predictions, preprocess_input
from scipy import ndimage, misc

data = np.load('FLICKR_test.npz')
print data.files

b1_input = data['input_test']
b1_target = data['target_test']

# b1_input = data['batch_1_input']
# b1_target = data['batch_1_target']
# b2_input = data['batch_2_input']
# b2_target = data['batch_2_target']

model = ResNet50(include_top=False, weights='imagenet')

print b1_input.shape

b1_features = []
b2_features = []
#img_path = './train/06984.jpg'
#img = image.load_img(img_path, target_size=(224, 224))

for i in range(b1_target.shape[0]):
    print i
    print b1_target[i]
    img = b1_input[i, :, :, :]
    image_resized = misc.imresize(img, (224, 224))
    x = image.img_to_array(image_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    print('type:', type(x))
    preds = model.predict(x)
    preds = preds.flatten()
    print('Predicted:', preds.shape)
    b1_features.append(preds)

# for i in range(b2_target.shape[0]):
#     print i
#     img = b2_input[i, :, :, :]
#     image_resized = misc.imresize(img, (224, 224))
#     x = image.img_to_array(image_resized)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     print('Input image shape:', x.shape)
#
#     print('type:', type(x))
#     preds = model.predict(x)
#     preds = preds.flatten()
#     print('Predicted:', preds.shape)
#     b2_features.append(preds)

np.savez('hogresnet_test', test_features = b1_features, test_target = b1_target)
#np.savez('hogresnet_k2batches', batch_1_features = b1_features, batch_2_features = b2_features, batch_1_target = b1_target, batch_2_target = b2_target)