# Hands-On Machine Learning with Scikit-Learn and TensorFlow

This repository is in reference to Chapter 2, End-to-End Machine Learning project. 

According to the author, the purpose of this chapter is to become familiarized with common tasks of Machine Learning.
This includes, retrieving data, analyze a given problem, analyze the data and gain insight, prepare and transform the data for Machine Learning algorithms, select a model and train it, fine-tune the model and finally present and maintain the solution.

The code in this repository is reproduced at best and slightly modified according to my study. It serves as a bookmark for future projects and as a reference to my Machine Learning portfolio.

Refer to the preface of the book if you want to reproduce this code for personal use. 

# Overview run.py 

### Preparing the project 

I use virtualenv locally to isolate the necessary Python packages. With ```pip freeze > requirements.txt``` every package in the environment was piped into a requirements file. The virtual environment has not been included in the source code.

### Overview of the problem

We need to build a model of the housing price data of California. This data consists of features such as population, median income, median housing price,... for a given district of California. From this model, we should be able to predict new housing prices for any other district. We will be working with pipelines, considering the prediction will be fed as an input to another Machine learning solution. 

### Constructing a model

This problem is clearly a regression problem (as opposed to a classification one), and more specifically, a multivariable regression problem, given the multiple features that our data is based upon. There is no motive for the model to constantly adapt to new data, meaning that a batch learning process will do for this project.  

In order to capture the error rate, we must find a performance measure. As mentioned in the book, a typical performance measure for regression problems is the Root Mean Square Error, or RMSE for short. It measures the standard deviation of the errors the system makes in its predictions and is based on the Gaussian distribution (68-95-99.7 rule). This means that for example, in the context of the housing prices, an RMSE value of 50000, means that 68% of all the predictions are close to 50000$ (1*sigma) of their actual label, 95% are close to 100000$ (2*sigma), and 99.7% to 150000$ (3*sigma). 

The RMSE function takes two inputs: 
- A column vector with instances of the data as its rows (which are row vectors) without their labels (so just the inputs without their output). 
- A hypothesis, which is the function that will predict the output for an instance. 
What RMSE does in the context of housing price is, predict the housing price for an instance, let's call it ŷ. Substract the actual housing price (let's call it y) for that instance, so ŷ - y. This is the error for that instance. We then sum the error of all instances, multiple it by the reciprocal of the amount of instances (call it m) and take the square root of it all. <b>Refer to page 37-39 for RMSE and MAE</b>

We are essentially just measuring the Euclidean distance between the label, and our own prediction for all instances. Refer to other measures such as MAE (Mean Absolute Error) as well. 

### Fetching the data

We fetch the tar file from the url mentioned in the book and extract it into a directory. The tar file contains the CSV file 
which represents the housing data. We need this data to train our model. Refer to the ```fetch_data([download_url, data_path, file_name])``` method. 
Afterwards, the ```load_csv([data_path, csv_name])``` function reads the data from the CSV file and returns a DataFrame 
object. 

### Analyzing the data

In the main method, we print the head (which is a 5x10 matrix) of this data just to visualize what the data looks 
like). Each row represents a district with 10 columns (attributes). We use ```.info()``` on the DataFrame object in order 
to retrieve more information about the columns and the amount of rows. For the sake not to pollute the output of the main 
method, we omit this in run.py. Recall that there are entries in which some attributes are missing (null). This is 
absolutely possible in real world data and we take care of this later on. Also notice that every column is numerical except 
the "ocean_proximity" attribute. This is a categorical attribute. We can visualize all the categories for this attribute by invoking 
```housing_data["ocean_proximity"].value_counts()``` on the DataFrame object. We can also use the ```describe()``` method on 
the object in order to get more information on the numerical attributes, such as the amount of entries for an attribute, the 
min, max and standard deviation values. For a more visual overview, we can use the ```hist()``` method on the DataFrame with 
matrplotlib in order to plot a histogram of the numerical attributes. A few things to note from the histograms is that some 
of the attributes were scaled and even capped. It is common for attributes to be preprocessed beforehand and this is not 
necessarily a problem. The histograms are also tail heavy, we will need to transform these later to more of a 
bell-shaped/Gaussian distribution. 

### Creating training and test data

In order to prevent overfitting our model, we have to split our data into sets. Recall that overfitting occurs when our 
model is generalized on our entire data, failing to perform well on new data. We use the ```train_test_split([data, 
test_size, random_state])``` function from the sklearn package for this utility. It will sample the data according to a seed 
(such as 42) and create a proportion (usually 20%) of the data into test data and the rest will be used as training data. We 
have used random sampling, but usually we have to consider that our test data must be representative of our whole 
dataset, otherwise we would have to deal with sampling bias (refer to stratified sampling). If we consider the fact that 
median income (one of the attributes of the housing data) is an important attribue to predict median housing prices, then 
the test set must be representative of the various categories of income in the whole dataset. This attribute is numerical 
and must be converted into categorical data. After we've done this, we will replace the ```train_test_split``` function with 
```split([data, column_to_split_by]``` using StratifiedShuffleSplit class from the sklearn package. 
