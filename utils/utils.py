import os
from typing import List
from functools import wraps
import time
import matplotlib.pyplot as plt
import numpy as np

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper



def concat_dataset(batched_dataset):
    timeseries = []
    labels = []
    for batch in batched_dataset:
        timeseries.extend(batch['data'])
        labels.extend(batch['labels'])
    return np.array(timeseries, dtype=np.float64), np.array(labels)


def dataset_batches(dataset, batch_size):
    values = dataset['data']
    labels = dataset['labels']

    batched_dataset = []

    current_batch = {'data': [], 'labels': []}
    for i, value in enumerate(values):
        if len(current_batch['data']) < batch_size:
            current_batch['data'].append(value)
            current_batch['labels'].append(labels[i])
        else:
            batched_dataset.append(current_batch)
            current_batch = {'data': [], 'labels': []}
    batched_dataset.append(current_batch)
    return batched_dataset

def seperate_data_labels(row: str):
    value, label = row.strip().split(',')
    return float(value), int(float(label))


def load_timeseries(dataset_path: str, file: List[str]):
    output = {'data': [], 'labels': []}
    """
    for file in files:
        with open(dataset_path + "/" + file, 'r') as f:
            for row in f.readlines():
                data, label = seperate_data_labels(row)
                output['data'].append(data)
                output['labels'].append(label)
        yield output
    """
    for f in os.listdir(dataset_path):
        if f == file:
            with open(dataset_path + "/" + file, 'r') as f:
                for row in f.readlines():
                    data, label = seperate_data_labels(row)
                    output['data'].append(data)
                    output['labels'].append(label)
            return output

def decide_anomaly(cpd_points):
    """
    The function takes as argument the change points that the signal changes its behaviour.
    To categorize a segment of a signal as an outlier, a point detected 
    as a change point should be at most 100 time units away from the previous one.
    """
    outlier_segments = []
    for i in range(1, len(cpd_points)):
        if cpd_points[i] - cpd_points[i-1] <= 100:
            outlier_segments.append((cpd_points[i-1], cpd_points[i]))
    return outlier_segments



def contains_number(tuples_list, number):
    for tup in tuples_list:
        if number in tup:
            return True
    return False

def refine_scores(scores, outlier_segments):
    """
    This function takes as arguments the list with the outlier scores 
    that the model produced and the outlier segments from the
    change point detection algorithm
    """
    new_scores = []
    for idx, score in enumerate(scores):
        if contains_number(outlier_segments, idx):
            new_scores.append(score + (1 - score)/2)
        else:
            new_scores.append(score - score/2)
    return new_scores


def transform_predictions_format(predictions: np.array):
    result = []
    for pred in predictions:
        if pred == 1:
            result.append(0)
        else:
            result.append(1)
    return result


def split_list(l, n):
    index_list = [None] + [i for i in range(1, len(l)) if l[i] - l[i - 1] > n] + [None]
    return [l[index_list[j - 1]:index_list[j]] for j in range(1, len(index_list))]



def plot_timeseries(values: List[float], labels: List[int]):
    timerange = [i for i in range(len(values))]
    normal = []
    time_normal = []
    outliers = []
    time_outliers = []
    for i, value in enumerate(values):
        if labels[i] == 1:
            outliers.append(value)
            time_outliers.append(timerange[i])
        else:
            normal.append(value)
            time_normal.append(timerange[i])

    segments_time_outliers = split_list(time_outliers, 2)
    segments_time_normal = split_list(time_normal, 2)

    segments_anomaly = []
    segments_normal = []
    for segment in segments_time_outliers:
        segments_anomaly.append(list(np.array(values)[segment]))
    for segment in segments_time_normal:
        segments_normal.append(list(np.array(values)[segment]))
    
    for i, segment in enumerate(segments_anomaly):
        plt.plot(segments_time_outliers[i], segment, color='red')
    for i, segment in enumerate(segments_normal):
        plt.plot(segments_time_normal[i], segment, color='blue')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Plot of a timeseries of the dataset')
    plt.show()