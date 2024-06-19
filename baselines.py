"""
In this python script we implemented the
two baseline methods for non-streaming outlier detection.
Specifically the two methods are the Isolation Forest and a DNN
"""


import os
import sys
import logging
from typing import List
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils.utils import plot_timeseries, transform_predictions_format, load_timeseries


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', level=logging.INFO)

            
        

def split_dataset(data: List, labels: List, percentage = 0.8):
    training_amount = int(percentage * len(data))
    data_train, data_test = data[:training_amount], data[training_amount:]
    labels_train, labels_test = labels[:training_amount], labels[training_amount:]
    return data_train, data_test, labels_train, labels_test


    

def isolation_forest(data, labels):
    X_train, X_test, y_train, y_test = split_dataset(data, labels)
    X_train = np.array(X_train).reshape(-1, 1)
    X_test = np.array(X_test).reshape(-1, 1)
    clf = IsolationForest(max_samples=100, random_state=0).fit(X_train)

    # evaluate train set
    predictions = clf.predict(X_train)
    predictions = transform_predictions_format(predictions)
    print('The accuracy score in train set is ', accuracy_score(np.array(y_train), predictions))

    # evaluate test set
    predictions = clf.predict(X_test)
    predictions = transform_predictions_format(predictions)
    print('The accuracy score in test set is ', accuracy_score(np.array(y_test), predictions))

def main():
    args = sys.argv
    if len(args) < 3:
        logging.warning('Write the baseline technique that you want to run on the timeseries')
        logging.info('Isolation Forest (isolation_forest) OR DNN (dnn)')
        logging.warning('Also the normality of the dataset that you want to make the benchmarking')
        
    if len(args) > 1:
        if args[1] == "isolation_forest":
            TIMESERIES_PATH = os.getcwd() + "/generated_ts_dataset/normality_" + args[2]
            files = os.listdir(TIMESERIES_PATH)
            gen = load_timeseries(TIMESERIES_PATH, files)
            for _ in range(len(files)):
                dataset = next(gen)
                # make a plot of the current dataset
                # plot_timeseries(dataset['data'], dataset['labels'])
                isolation_forest(dataset['data'], dataset['labels'])
        elif args[1] == "dnn":
            pass
    else:
        pass
        




if __name__ == '__main__':
    main()


