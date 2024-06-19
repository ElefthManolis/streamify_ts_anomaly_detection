"""
The script below genarate dataset for each nornmality level 
based on the project description
"""

import os
import random
import shutil

PUBLIC_DATASET_PATH = os.getcwd() + "/data/TSB-UAD-Public"


def merge_files(filepath1, filepath2, filepath3 = None, destination_path = None):
    # open and parse the data from file1
    file1 = open(filepath1, 'r')
    content1 = file1.read()
    file1.close()

    # open and parse the data from file2
    file2 = open(filepath2, 'r')
    content2 = file2.read()
    file2.close()

    filename_1 = filepath1.split('/')[-1]
    filename_2 = filepath2.split('/')[-1]
    merged_filename = "merged_" + filename_1 + "_" + filename_2 + ".txt"

    if filepath3:
        # open and parse the data from file3
        file3 = open(filepath3, 'r')
        content3 = file3.read()
        file3.close()
        filename_3 = filepath3.split('/')[-1]
        merged_filename = "merged_" + filename_1 + "_" + filename_2 + "_" + filename_3 + ".txt"

    destination_file = destination_path + "/" + merged_filename
    destination_file = open(destination_file, 'w')
    destination_file.write(content1 + content2)
    if filepath3:
        destination_file.write(content3)
    destination_file.close()
    


def main():
    datasets = os.listdir(PUBLIC_DATASET_PATH)
    # remove DS_Store folder from my list
    if '.DS_Store' in datasets:
        datasets.remove('.DS_Store')
    random_datasets = random.sample(datasets, k=6)
    print(random_datasets)

    # Build random dataset with Normality 1
    normality_1_path_src = PUBLIC_DATASET_PATH + "/" + random_datasets[0]
    normality_1_path_dst = os.getcwd() + "/generated_ts_dataset/normality_1"
    if len(os.listdir(normality_1_path_dst)) == 0:  # check if the destination folder isn't empty
        for file in os.listdir(normality_1_path_src):
            # copy the timeseries with propability 0.5
            if random.random() > 0.5:
                shutil.copyfile(normality_1_path_src + "/" + file, normality_1_path_dst + "/" + file)
    
    
    

    # Build random dataset with Normality 2
    normality_2_path_src_1 = PUBLIC_DATASET_PATH + "/" + random_datasets[1]
    normality_2_path_src_2 = PUBLIC_DATASET_PATH + "/" + random_datasets[2]
    normality_2_path_dst = os.getcwd() + "/generated_ts_dataset/normality_2"
    if len(os.listdir(normality_2_path_dst)) == 0:
        for file1, file2 in zip(os.listdir(normality_2_path_src_1), os.listdir(normality_2_path_src_2)):
            filepath_1 = normality_2_path_src_1 + "/" + file1
            filepath_2 = normality_2_path_src_2 + "/" + file2
            # concat two different timeseries with propobility 0.5
            if random.random() > 0.5:
                merge_files(filepath_1, filepath_2, destination_path=normality_2_path_dst)



    # Build random dataset with Normality 3
    normality_3_path_src_1 = PUBLIC_DATASET_PATH + "/" + random_datasets[3]
    normality_3_path_src_2 = PUBLIC_DATASET_PATH + "/" + random_datasets[4]
    normality_3_path_src_3 = PUBLIC_DATASET_PATH + "/" + random_datasets[5]
    normality_3_path_dst = os.getcwd() + "/generated_ts_dataset/normality_3"
    if len(os.listdir(normality_3_path_dst)) == 0:
        for file1, file2, file3 in zip(os.listdir(normality_3_path_src_1), os.listdir(normality_3_path_src_2), os.listdir(normality_3_path_src_3)):
            filepath_1 = normality_3_path_src_1 + "/" + file1
            filepath_2 = normality_3_path_src_2 + "/" + file2
            filepath_3 = normality_3_path_src_3 + "/" + file3
            # concat three different timeseries with propobility 0.5
            if random.random() > 0.5:
                merge_files(filepath_1, filepath_2, filepath_3, normality_3_path_dst)


if __name__ == '__main__':
    main()