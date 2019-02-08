import os
import tarfile
import numpy as np
from six.moves import urllib
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

DOWNLOAD_URL = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"
DATA_PATH = "data"
FILE_NAME = "housing.tgz"
CSV_NAME = "housing.csv"

def fetch_data(download_url=DOWNLOAD_URL, data_path=DATA_PATH, file_name=FILE_NAME):
    if not os.path.isdir(data_path): 
        os.makedirs(data_path)
    full_path = os.path.join(data_path, file_name)    
    urllib.request.urlretrieve(url=download_url, filename=full_path)
    file = tarfile.open(full_path)
    file.extractall(path=data_path)
    file.close()
    os.remove(full_path)

def load_csv(data_path=DATA_PATH, csv_name=CSV_NAME):
    csv_path= os.path.join(data_path, csv_name)
    return pd.read_csv(csv_path)

if __name__ == "__main__": 
    fetch_data()
    housing_data = load_csv()
    #print(housing_data.head()) 
    #print(housing_data.info())
    #print(housing_data["ocean_proximity"].value_counts())
    #print(housing_data.describe())
    #housing_data.hist(bins=50, figsize=(20,15))
    #plt.show()
    #train_set, test_set = train_test_split(housing_data, test_size=0.2, random_state=42)
    housing_data["income_cat"] = np.ceil(housing_data["median_income"]/1.5)
    housing_data["income_cat"].where(housing_data["income_cat"] < 5, 5.0, inplace=True)
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) 
    for train_index, test_index in split.split(housing_data, housing_data["income_cat"]):
        strat_train_set = housing_data.loc[train_index]
        strat_test_set = housing_data.loc[test_index]
    for set in (strat_train_set, strat_test_set): 
        set.drop(["income_cat"], axis=1, inplace=True)
