
# Import the Important Modules & Functions
from model_monitering import read_dataset , is_dataset_balance , convert_matrix , preprocessing , split_dataset , model_development
import mlflow



# Run the Main  (For the Runing Project)
if __name__ == "__main__":
    
    


    # Define the DATASET PATH
    DATASET_PATH = "../Dataset/fraud_detection_dataset.csv"
    dataset      = read_dataset(DATASET_PATH)
    
    # Display the Count of the Class Label
    firstCount , secondCount = is_dataset_balance(dataset)
    # # Display the Count of the First & Second Class Label
    # print(f"\nHere is the Count of First Class Label  : {firstCount}")
    # print(f"Here is the Count of Second Class Label : {secondCount}")

    # Convert the Dataset into X & y Numpy Array Matrix
    X , y = convert_matrix(dataset)
    # # Display the Shape of X & y Matrix
    # print(f"\nHere is the Shape of the X Matrix : {X.shape}")
    # print(f"Here is the Shape of the Y Matrix : {y.shape}")

    # Pre-processing the Dataset
    X , y = preprocessing(X , y)

    # Split the Dataset into Training & Testing 
    x_train , x_test , y_train , y_test = split_dataset(X , y)
    # # Display the Shape of the Training and Testing 
    # print(f"Here is the Shape of the X Train : {x_train.shape}")
    # print(f"Here is the Shape of the Y Train : {y_train.shape}")
    # print(f"Here is the Shape of the X Test  : {x_test.shape}")
    # print(f"Here is the Shape of the Y Test  : {y_test.shape}")

    # Development the Model in the Training & Testing Dataset
    models_acc = model_development(x_train , x_test , y_train , y_test , dataset)
    
    # Display the Accuracy
    print(f"Here is the Accuracy of the Models : {models_acc}")


