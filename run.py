import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt

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
    
