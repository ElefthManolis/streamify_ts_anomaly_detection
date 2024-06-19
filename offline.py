import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import logging
from TSB_UAD.models.distance import Fourier
from TSB_UAD.models.feature import Window
from TSB_UAD.utils.slidingWindows import find_length,plotFig
from sklearn.preprocessing import MinMaxScaler

from TSB_UAD.models.iforest import IForest
from TSB_UAD.models.lstm import lstm


from utils.utils import load_timeseries, timeit

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', level=logging.INFO)



# Isolation Forest
@timeit
def isolation_forest(timeseries, labels, filename):
    # convert the timeseries from list to numpy array
    timeseries = np.array(timeseries, dtype=np.float64)
    # convert also the labels from list to numpy array
    labels = np.array(labels)
    

    slidingWindow = find_length(timeseries)
    X_data = Window(window = slidingWindow).convert(timeseries).to_numpy()

    data_train = timeseries[:int(0.1*len(timeseries))]
    data_test = timeseries

    X_train = Window(window = slidingWindow).convert(data_train).to_numpy()
    X_test = Window(window = slidingWindow).convert(data_test).to_numpy()



    modelName='IForest'
    clf = IForest(n_jobs=-1) # use all the cores to accelerate the procedure (n_jobs=-1)
    x = X_data
    clf.fit(x)
    score = clf.decision_scores_
    

    # Post processing
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))

    # save result as figure
    plotFig(timeseries, labels, score, slidingWindow, fileName=filename, modelName=modelName)
    plt.savefig(modelName+'_'+filename+'.png')
    plt.close()
    


# dnn method with lstm neural network
@timeit
def dnn(timeseries, labels, filename):
    timeseries = np.array(timeseries, dtype=np.float64)
    labels = np.array(labels)
    slidingWindow = find_length(timeseries)

    data_train = timeseries[:int(0.1*len(timeseries))]
    data_test = timeseries

    modelName='LSTM'
    clf = lstm(slidingwindow = slidingWindow, predict_time_steps=1, epochs = 10, patience = 5, verbose=0)
    
    clf.fit(data_train, data_test)
    measure = Fourier()
    measure.detector = clf
    measure.set_param()
    clf.decision_function(measure=measure)
            
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()

    plotFig(timeseries, labels, score, slidingWindow, fileName=filename, modelName=modelName)
    plt.savefig(modelName+'_'+filename+'.png')
    plt.close()

def main():
    args = sys.argv
    if len(args) < 3:
        logging.warning('Write the baseline technique that you want to run on the timeseries')
        logging.info('Isolation Forest (isolation_forest) OR DNN (dnn)')
        logging.warning('Also the normality of the dataset that you want to make the benchmarking')
    if len(args) > 1:
        TIMESERIES_PATH = os.getcwd() + "/generated_ts_dataset/normality_" + args[2]
        files = os.listdir(TIMESERIES_PATH)
        #gen = load_timeseries(TIMESERIES_PATH, files)
        if args[1] == "isolation_forest":
            for i in range(len(files)):
                dataset = load_timeseries(TIMESERIES_PATH, files[i])
                isolation_forest(dataset['data'], dataset['labels'], files[i])
        elif args[1] == "dnn":
            for i in range(len(files)):
                dataset = load_timeseries(TIMESERIES_PATH, files[i])
                dnn(dataset['data'], dataset['labels'], files[i])




if __name__ == '__main__':
    main()