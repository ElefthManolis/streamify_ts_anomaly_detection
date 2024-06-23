# streamify_ts_anomaly_detection

## Installation

Clone the project

```bash
git clone git@github.com:ElefthManolis/streamify_ts_anomaly_detection.git
```

Inside the project clone the TSB-UAD repository

```bash
git clone git@github.com:TheDatumOrg/TSB-UAD.git
```

and follow the installation instructions of the repo


## Generate daatsets
Create a folder **data** and download the TSB-UAD-Public dataset.

In order to generate the datasets with the different normalities you have to run the python script below
```
python3 generate_datasets.py 
```


## Run the experiments

In order to run the offline baseline algorithms:

```
python3 offline_baselines.py <MODEL> <NORMALITY>
```

In order to run the **SAND** algorithm you have to run:
```
python3 sand.py <NORMALITY>
```

Finally, to run the modified offline algorithms but in online mode:
```
python3 online.py <MODEL> <NORMALITY> <VARIANT> <BATCH_SIZE>
```

where in the **MODEL** argument you have to choose between *isolation_forest* or *dnn*, and in the **NORMALITY** argument between 1, 2 or 3 (normality of the generated dataset).\
The **VARIANT** argument refers to the variant of the algorithms (1 or 2) and the **BATCH_SIZE** you can set the batch size for the online processing of the timeseries.