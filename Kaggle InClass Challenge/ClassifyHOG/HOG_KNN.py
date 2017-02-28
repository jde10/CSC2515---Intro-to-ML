import numpy as np
from sklearn import metrics
from sklearn import svm

train_data = np.load('HOG_features.npz')
test_data = np.load('HOG_features_test.npz')

print train_data.files
print test_data.files

train_input = train_data['batch_1_HOG']
train_target = train_data['batch_1_target']
val_input = train_data['batch_2_HOG']
val_target = train_data['batch_2_target']

test_input = test_data['test_HOG']
test_target = np.array(range(1, 971))

train_input = train_input.reshape(train_input.shape[0], 128*128)
val_input = val_input.reshape(val_input.shape[0], 128*128)
test_input = test_input.reshape(test_input.shape[0], 128*128)
print train_input.shape

model = svm.SVC(decision_function_shape='ovo')
model.fit(train_input, train_target)

#make predictions
pred_val = model.predict(val_input)
print(metrics.classification_report(val_target, pred_val))
print(metrics.confusion_matrix(val_target, pred_val))

pred_test = model.predict(test_input)
print(metrics.classification_report(test_target, pred_test))
print(metrics.confusion_matrix(test_target, pred_test))
count = 0
for i in range(val_target.shape[0]):
    print pred_val
    if val_target[i] == pred_val[i]:
        count+=1

acc = count/val_target.shape[0]
print('Accuracy: %.2f' % acc)

#test_target is id
#pred_test is the prediction

sub = [','.join((id,pred)) for id, pred in zip(test_target, pred_test)]
sub = '\n'.join(sub)
with open('submission_hog_svm.csv', 'w') as fn:
    fn.write(sub)
