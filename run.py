import os
import tarfile
from six.moves import urllib

DOWNLOAD_URL = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"
DATA_PATH = "data"
FILE_NAME = "housing.tgz"

def fetch_data(download_url=DOWNLOAD_URL, data_path=DATA_PATH, file_name=FILE_NAME):
    if not os.path.isdir(data_path): 
        os.makedirs(data_path)
    full_path = os.path.join(data_path, file_name)    
    urllib.request.urlretrieve(url=download_url, filename=full_path)
    file = tarfile.open(full_path)
    file.extractall(path=data_path)
    file.close()
    os.remove(full_path)

if __name__ == "__main__": 
    fetch_data()
     
