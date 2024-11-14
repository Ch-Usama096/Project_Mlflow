# Import the Important Modules
import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import mlflow


# Ingore the Warnings
warnings.simplefilter(action='ignore', category=DataConversionWarning)


# Create the Fucntions for the Reading Dataset
def read_dataset(DATASET_PATH):
    # Read the Dataset 
    dataset = pd.read_csv(DATASET_PATH)

    # Return the Dataset
    return dataset


# Create the Function for Finding the Information about the Dataset Such as (Shape , Missing Values , Duplictaed Data)
def info_dataset(dataset):
    # Display the Shape of the Dataset
    print(f"\nHere is the Shape of the Dataset : {dataset.shape}")

    # Check the Missing Values in the Dataset
    missingValues = dataset.isnull().sum()
    print(f"Here is the Missing Values of each Colm : \n{missingValues}")

    # Check the Duplicated Row in the Dataset
    duplicatedRow = dataset.duplicated().sum()
    print(f"Here is the Total Duplicated Row in the Dataset : {duplicatedRow}")


# Create the Function for checking the Balance or Not Balance Dataset
def is_dataset_balance(dataset):
    # Get the Count of Each Class Label
    countFirst , countSecond = dataset["Fraud"].value_counts()

    # # Display the Count of the First & Second Class Label
    # print(f"Here is the Count of First Class Label  : {countFirst}")
    # print(f"Here is the Count of Second Class Label : {countSecond}")

    # Return the Count Label
    return countFirst , countSecond


# Create the Function for Conmverting the Dataset into Dependent & Independent Matrix
def convert_matrix(dataset):
    # Convert the Dataset into Dependent & Independent Matrix
    X = dataset.iloc[:,:-1].values
    y = dataset.iloc[:,-1].values

    # # Display the Shape of X & y Matrix
    # print(f"Here is the Shape of the X Matrix : {X.shape}")
    # print(f"Here is the Shape of the Y Matrix : {y.shape}")

    # Return the X & y Matrix
    return X , y


# Create the Function For the Preprocessing of the Dataset
def preprocessing(X , y):
    # Preprocessing the Dataset

    # Convert the Categorical Data into Numerical Data
    encoder   = LabelEncoder() # Create the Object of LabelEncoder
    cat_index = [0 , 3] 
    for index in cat_index:
        X[:,[index]] = encoder.fit_transform(X[:,[index]]).reshape(-1,1)
    
    
    # Normalize the Data (Convert Numerical Data into Range (0-1))
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Return the X & y
    return X , y


# Create the Function for Splitting the Dataset into Training & Testing
def split_dataset(X , y):
    # Split the Dataset into Training & Testing 
    x_train , x_test , y_train , y_test = train_test_split(X , y , test_size = 0.25 , random_state = 42)

    # # Display the Shape of the Training and Testing 
    # print(f"Here is the Shape of the X Train : {x_train.shape}")
    # print(f"Here is the Shape of the Y Train : {y_train.shape}")
    # print(f"Here is the Shape of the X Test  : {x_test.shape}")
    # print(f"Here is the Shape of the Y Test  : {y_test.shape}")

    # Return the training and Testing Data
    return x_train , x_test , y_train , y_test




# Create the Functions for Appling the Model
def model_development(x_train , x_test , y_train , y_test):
    
    # Track the Model Training  & Save in the MLFLOW
    with mlflow.start_run():

    # with mlflow.start_run(run_name = "logistic Regression"):

        # Create the Model Object
        lg = LogisticRegression()
        
        # Fit the Model in the Training Dataset
        lg.fit(x_train , y_train)

        # Predict the Result
        prediction = lg.predict(x_test)
        
        acc = accuracy_score(y_test , prediction)

        # Save the Model Metrics
        mlflow.log_metric("Accuracy Score" , acc)


