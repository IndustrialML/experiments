# Spark configuration and packages specification. The dependencies defined in
# this file will be automatically provisioned for each run that uses Spark.

import sys
import os
import pickle

from sklearn.datasets import fetch_mldata
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np


class mockClf:
    def predict(self,input):
        return 0




# This is how you log scalar metrics
# logger.log("MyMetric", value)

# Create the outputs folder - save any outputs you want managed by AzureML here
os.makedirs('./outputs', exist_ok=True)



# train a logistic regression model on the training set
clf = mockClf()



print("")
print("==========================================")
print("Serialize and deserialize using the outputs folder.")
print("")

# serialize the model on disk in the special 'outputs' folder
print ("Export the model to model.pkl")
f = open('./outputs/model.pkl', 'wb')
pickle.dump(clf, f)
f.close()



# load the model back from the 'outputs' folder into memory
print("Import the model from model.pkl")
f2 = open('./outputs/model.pkl', 'rb')
clf2 = pickle.load(f2)


# score on the new sample
pred = clf2.predict([0,1])
print('Predicted class is {}'.format(pred))
