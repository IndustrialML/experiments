# Spark configuration and packages specification. The dependencies defined in
# this file will be automatically provisioned for each run that uses Spark.

import sys
import os
import argparse
import pickle
import struct
import array
from sklearn.datasets import fetch_mldata
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

from azureml.logging import get_azureml_logger

def load_mnist(dataset="training", digits=np.arange(10), path="C:/Users/lema/Documents/projects/Azure_ML/data", size = 60000):
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = array.array("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = array.array("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = size #int(len(ind) * size/100.)
    images = np.zeros((N, rows, cols), dtype=np.uint8)
    labels = np.zeros((N, 1), dtype=np.int8)
    for i in range(N): #int(len(ind) * size/100.)):
        images[i] = np.array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ])\
            .reshape((rows, cols))
        labels[i] = lbl[ind[i]]
    labels = [label[0] for label in labels]
    return np.array(images).reshape(-1,784), np.array(labels)


# initialize the logger
logger = get_azureml_logger()

# add experiment arguments
parser = argparse.ArgumentParser()
# parser.add_argument('--arg', action='store_true', help='My Arg')
args = parser.parse_args()
print("Args:%s"%args)

# This is how you log scalar metrics
# logger.log("MyMetric", value)

# Create the outputs folder - save any outputs you want managed by AzureML here
os.makedirs('./outputs', exist_ok=True)

# load dataset -> features,labels
X_train, Y_train = load_mnist("training")
X_test, Y_test = load_mnist("testing")
print ('MNIST train_images dataset shape: {}'.format(X_train.shape))
print ('MNIST train_labels dataset shape: {}'.format(X_test.shape))
print ('MNIST test_images dataset shape: {}'.format(Y_train.shape))
print ('MNIST test_labels dataset shape: {}'.format(Y_test.shape))



# train a logistic regression model on the training set
clf = RandomForestClassifier(n_estimators = 500, max_features=28, criterion="gini")
print (clf)

clf.fit(X_train,Y_train)

# evaluate the test set
accuracy = clf.score(X_test, Y_test)
print ("Accuracy is {}".format(accuracy))

# log accuracy which is a single numerical value
logger.log("Accuracy", accuracy)



print("")
print("==========================================")
print("Serialize and deserialize using the outputs folder.")
print("")

# serialize the model on disk in the special 'outputs' folder
print ("Export the model to model.pkl")
f = open('./outputs/model_500.pkl', 'wb')
pickle.dump(clf, f)
f.close()



# load the model back from the 'outputs' folder into memory
print("Import the model from model.pkl")
f2 = open('./outputs/model.pkl', 'rb')
clf2 = pickle.load(f2)

# predict on a new sample
X_new = np.array([X_test[0]])
print ('New sample: {}'.format(X_new))
print ('New sample shape: {}'.format(X_new.shape))

# score on the new sample
pred = clf2.predict(X_new)
print('Predicted class is {}'.format(pred))
