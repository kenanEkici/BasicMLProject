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

In order to capture the error rate, we must find a performance measure. As mentioned in the book, a typical performance measure for regression problems is the Root Mean Square Error, or RMSE for short. It measures the standard deviation of the errors the system makes in its predictions and is based on the Gaussian distribution (68-95-99.7 rule).
It takes two inputs: 
- A column vector with instances of the data as its rows (which are row vectors) without their labels (so just the inputs without their output). 
- A hypothesis, which is the function that will predict the output for an instance. 
What RMSE does in the context of housing price is, predict the housing price for an instance, let's call it ŷ. Substract the actual housing price (let's call it y) for that instance, so ŷ - y. This is the error for that instance. We then sum the error of all instances, multiple it by the reciprocal of the amount of instances (call it m) and take the square root of it all. <b>Refer to page 37-39 for this measure</b>

We are essentially just measuring the Euclidean distance between the label, and our own prediction. Refer to other measures such as MAE (Mean Absolute Error) as well. 

### Fetching the data

We fetch the tar file from the url mentioned in the book and extract it into a directory. The tar file contains the CSV file which represents the housing data. We need this data to train our model. Refer to the ```fetch_data([download_url, data_path, file_name)``` method. 


