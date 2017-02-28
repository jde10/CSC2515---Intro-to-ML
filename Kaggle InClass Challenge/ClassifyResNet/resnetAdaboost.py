import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

train_data = np.load('hogresnet_k2batches.npz')
test_data = np.load('hogresnet_test.npz')

print train_data.files
print test_data.files

train_input = train_data['batch_1_features']
train_target = train_data['batch_1_target']
val_input = train_data['batch_2_features']
val_target = train_data['batch_2_target']

test_input = test_data['test_features']
test_target = test_data['test_target']


#model = AdaBoostClassifier(n_estimators = 100)
#model = RandomForestClassifier(n_estimators = 1000)

model = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(model, train_input, train_target)
print scores.mean()

model = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(model, train_input, train_target)
print scores.mean()

model = ExtraTreesClassifier(n_estimators=1000, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(model, train_input, train_target)
print scores.mean()

model.fit(train_input, train_target)

#make predictions
pred_val = model.predict(val_input)
print(metrics.classification_report(val_target, pred_val))
print(metrics.confusion_matrix(val_target, pred_val))

pred_test = model.predict(test_input)


#test_target is id
#pred_test is the prediction

sub = [','.join((id,pred)) for id, pred in zip(test_target, pred_test)]
sub = '\n'.join(sub)
with open('submission_ensemble_extreme.csv', 'w') as fn:
    fn.write(sub)
