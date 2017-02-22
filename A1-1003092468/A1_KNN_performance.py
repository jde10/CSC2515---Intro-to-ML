import numpy as np
from run_knn import run_knn
from l2_distance import l2_distance
import matplotlib.pyplot as plot


# Load files from dir
mnist_train = np.load('/Users/JDLVF/Documents/UofT/Fall 2016/CSC411 Machine Learning/Assignment 1/hw1_code_question/mnist_train.npz')
mnist_valid = np.load('/Users/JDLVF/Documents/UofT/Fall 2016/CSC411 Machine Learning/Assignment 1/hw1_code_question/mnist_valid.npz')
mnist_test = np.load('/Users/JDLVF/Documents/UofT/Fall 2016/CSC411 Machine Learning/Assignment 1/hw1_code_question/mnist_test.npz')

# variables
k = [1, 3, 5, 7, 9]
train_data = mnist_train.f.train_inputs
train_labels = mnist_train.f.train_targets
valid_data = mnist_valid.f.valid_inputs
valid_labels = mnist_valid.f.valid_targets
test_data= mnist_test.f.test_inputs
test_labels = mnist_test.f.test_targets
c_rate = np.zeros(shape=len(k))

# run knn on validation set

for i in range(0, len(k)):
    if i == 0:
        valid_pred= run_knn(k[i],train_data, train_labels, valid_data)
        incorrect = np.sum ( np.absolute ( np.subtract (valid_labels, valid_pred) ) )
        c_rate[i] = (len(valid_labels) - incorrect) / len(valid_labels)
    else:
        temp = run_knn(k[i],train_data, train_labels, valid_data)
        valid_pred= np.concatenate((valid_pred, temp), axis=1)
        incorrect = np.sum ( np.absolute ( np.subtract (valid_labels, temp) ) )
        c_rate[i] = (len(valid_labels) - incorrect) / len(valid_labels)
        
# result on validation c_rate = [0.94,  0.98,  0.98,  0.98,  0.96]

# plotting the results for validation set

plot.figure(1)
plot.plot(k, c_rate, color="blue", linewidth=2.5, linestyle="-", label="Classification Rate")
plot.legend(loc='lower right')
plot.xlabel('k')
plot.ylabel('Correct Predictions / Total Data Points')
plot.title('Classification Rate in Validation Set')
plot.grid(True)
plot.axis([1, 9, 0.93, 1])
plot.show()

# variable reset for test set
del k, incorrect, c_rate, temp
k = [3, 5, 7]
c_rate = np.zeros(shape=len(k))

# run knn on test set
for i in range(0, len(k)):
    if i == 0:
        test_pred= run_knn(k[i],train_data, train_labels, test_data)
        incorrect = np.sum ( np.absolute ( np.subtract (test_labels, test_pred) ) )
        c_rate[i] = (len(test_labels) - incorrect) / len(test_labels)
    else:
        temp = run_knn(k[i],train_data, train_labels, test_data)
        test_pred= np.concatenate((test_pred, temp), axis=1)
        incorrect = np.sum ( np.absolute ( np.subtract (test_labels, temp) ) )
        c_rate[i] = (len(test_labels) - incorrect) / len(test_labels)
        
# result on test c_rate = [0.98,  0.98,  0.92]

# plotting the results for test set

plot.figure(2)
plot.plot(k, c_rate, color="blue", linewidth=2.5, linestyle="-", label="Classification Rate")
plot.legend(loc='lower right')
plot.xlabel('k')
plot.ylabel('Correct Predictions / Total Data Points')
plot.title('Classification Rate on Test Set')
plot.grid(True)
plot.axis([1, 9, 0.91, 1])
plot.show()
