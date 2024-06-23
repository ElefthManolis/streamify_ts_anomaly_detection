import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from TSB_UAD.models.distance import Fourier
from TSB_UAD.models.feature import Window
from TSB_UAD.utils.slidingWindows import find_length,plotFig, printResult
from sklearn.preprocessing import MinMaxScaler

from TSB_UAD.models.iforest import IForest
from TSB_UAD.models.lstm import lstm

from utils.utils import (load_timeseries, dataset_batches, 
                        concat_dataset, decide_anomaly, refine_scores,
                        timeit)


import ruptures as rpt # library for change point detection



@timeit
def variant1(batched_dataset, filename, model):
    """
    Streamify static methods
    """

    timeseries, labels = concat_dataset(batched_dataset)
    
    if model == 'IForest':
        modelName='IForest'
        clf = IForest(n_jobs=-1) # use all the cores to accelerate the procedure (n_jobs=-1)
        scores = []
        # iterate throw batches in the entire signal
        for batch in batched_dataset:
            # convert the timeseries from list to numpy array
            batch_ts = np.array(batch['data'], dtype=np.float64)
            # convert also the labels from list to numpy array
            batch_labels = np.array(batch['labels'])


            slidingWindow = find_length(batch_ts)
            X_data = Window(window = slidingWindow).convert(batch_ts).to_numpy()

            data_train = batch_ts[:int(0.1*len(batch_ts))]
            data_test = batch_ts

            
            x = X_data
            clf.fit(x)
            score = clf.decision_scores_
            
            # Post processing
            score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
            score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
            scores.append(score)

        scores = np.array(list(itertools.chain.from_iterable(scores))) # merge all the score lists

        plotFig(timeseries, labels, scores, slidingWindow, fileName=filename, modelName=modelName)
        plt.savefig('online_'+modelName+'_'+filename+'.png')
        plt.close()
        
    elif model == 'lstm':

        modelName='LSTM'
        scores = []
        slidingWindow = 7
        clf = lstm(slidingwindow = slidingWindow, predict_time_steps=1, epochs = 5, patience = 5, verbose=0)
        # iterate throw batches in the entire signal
        for batch in batched_dataset:

            batch_ts = np.array(batch['data'], dtype=np.float64)
            batch_labels = np.array(batch['labels'])
                     
            data_train = batch_ts[:int(0.1*len(batch_ts))]
            data_test = batch_ts

            
            clf.fit(data_train, data_test)
            measure = Fourier()
            measure.detector = clf
            measure.set_param()
            clf.decision_function(measure=measure)
                    
            score = clf.decision_scores_
            score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
            scores.append(score)

        scores = np.array(list(itertools.chain.from_iterable(scores))) # merge all the score lists

        plotFig(timeseries, labels, scores, slidingWindow, fileName=filename, modelName=modelName)
        plt.savefig('online_'+modelName+'_'+filename+'.png')
        plt.close()


@timeit
def variant2(batched_dataset, filename, model):
    """
    Implementation of the variant 2
    for the Isolation Forest and the LSTM architecture
    """

    timeseries, labels = concat_dataset(batched_dataset)

    if model == 'IForest':
        modelName='IForest'
        clf = IForest(n_jobs=-1) # use all the cores to accelerate the procedure (n_jobs=-1)
        scores = []
        # iterate throw batches in the entire signal
        for batch in batched_dataset:
            # convert the timeseries from list to numpy array
            batch_ts = np.array(batch['data'], dtype=np.float64)
            # convert also the labels from list to numpy array
            batch_labels = np.array(batch['labels'])

            # detect change point in the batched signal
            algo = rpt.Pelt(model="rbf").fit(batch_ts)
            result = algo.predict(pen=10)
            outlier_segments = decide_anomaly(result)


            slidingWindow = find_length(batch_ts)
            X_data = Window(window = slidingWindow).convert(batch_ts).to_numpy()

            
            x = X_data
            clf.fit(x)
            score = clf.decision_scores_
            
            # Post processing
            score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
            score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
            
            score = refine_scores(score, outlier_segments)
            # print(score[[*map(lambda x: x - 1, result)]])
            scores.append(score)

        scores = np.array(list(itertools.chain.from_iterable(scores))) # merge all the score lists

        plotFig(timeseries, labels, scores, slidingWindow, fileName=filename, modelName=modelName)
        plt.savefig('online_'+modelName+'_'+filename+'.png')
        plt.close()
    elif model == 'lstm':

        modelName='LSTM'
        scores = []
        slidingWindow = 7
        clf = lstm(slidingwindow = slidingWindow, predict_time_steps=1, epochs = 5, patience = 5, verbose=0)
        # iterate throw batches in the entire signal
        for batch in batched_dataset:

            batch_ts = np.array(batch['data'], dtype=np.float64)
            batch_labels = np.array(batch['labels'])

            # detect change point in the batched signal
            algo = rpt.Pelt(model="rbf").fit(batch_ts)
            result = algo.predict(pen=10)
            outlier_segments = decide_anomaly(result)
                     
            data_train = batch_ts[:int(0.1*len(batch_ts))]
            data_test = batch_ts

            
            clf.fit(data_train, data_test)
            measure = Fourier()
            measure.detector = clf
            measure.set_param()
            clf.decision_function(measure=measure)
                    
            score = clf.decision_scores_
            score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()

            score = refine_scores(score, outlier_segments)
            scores.append(score)

        scores = np.array(list(itertools.chain.from_iterable(scores))) # merge all the score lists

        plotFig(timeseries, labels, scores, slidingWindow, fileName=filename, modelName=modelName)
        plt.savefig('online_'+modelName+'_'+filename+'.png')
        plt.close()



def main():
    """
    arg1: model (isolation_forest or dnn)
    arg2: dataset normality (1, 2 or 3)
    arg3: variant (1, 2)
    arg4: batch size of the splitted dataset
    """
    args = sys.argv
    if len(args) > 1:
        TIMESERIES_PATH = os.getcwd() + "/generated_ts_dataset/normality_" + args[2]
        files = os.listdir(TIMESERIES_PATH)
        if args[1] == "isolation_forest":
            for i in range(len(files)):
                dataset = load_timeseries(TIMESERIES_PATH, files[i])
                batched_dataset = dataset_batches(dataset, int(args[4]))
                if int(args[3]) == 1:
                    # Variant 1
                    variant1(batched_dataset, files[i], 'IForest')
                elif int(args[3]) == 2:
                    # Variant 2
                    variant2(batched_dataset, files[i], 'IForest')
        elif args[1] == "dnn":
            for i in range(len(files)):
                dataset = load_timeseries(TIMESERIES_PATH, files[i])
                batched_dataset = dataset_batches(dataset, int(args[4]))
                if int(args[3]) == 1:
                    # Variant 1
                    variant1(batched_dataset, files[i], 'lstm')
                elif int(args[3]) == 2:
                    # Variant 2
                    variant2(batched_dataset, files[i], 'lstm')



if __name__ == '__main__':
    main()