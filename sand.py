import os
import sys
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings 


from TSB_UAD.utils.slidingWindows import find_length, plotFig
from sklearn.preprocessing import MinMaxScaler

from utils.utils import load_timeseries, timeit

from TSB_UAD.models.sand import SAND

warnings.filterwarnings('ignore') 


@timeit
def sand(timeseries, labels, filename):
    timeseries = np.array(timeseries, dtype=np.float64)
    labels = np.array(labels)
    slidingWindow = find_length(timeseries)
    if slidingWindow > 100:
        slidingWindow = 100
    
    modelName='SAND (online)'
    clf = SAND(pattern_length=slidingWindow,subsequence_length=4*(slidingWindow))
    x = timeseries
    clf.fit(x,online=True,alpha=0.5,init_length=1000,batch_size=2000,verbose=True,overlaping_rate=int(4*slidingWindow))
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    plotFig(timeseries, labels, score, slidingWindow, fileName=filename, modelName=modelName)
    plt.savefig(modelName+'_'+filename+'.png')
    plt.close()

def main():
    args = sys.argv
    TIMESERIES_PATH = os.getcwd() + "/generated_ts_dataset/normality_" + args[1]
    files = os.listdir(TIMESERIES_PATH)
    for i in tqdm(range(len(files))):
        dataset = load_timeseries(TIMESERIES_PATH, files[i])
        sand(dataset['data'], dataset['labels'], files[i])

if __name__ == '__main__':
    main()