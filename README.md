# Car Value Estimator In TensorFlow

## Introduction
This project aims to develop a linear regression model to estimate the value of a vehicle based on various features such as mileage, age, engine power, and transmission type. The CarDekho Vehicle Dataset was used for this project, with data cleaned in Google Sheets. The model was implemented using Python and TensorFlow. For more information visit [https://bchharaw.github.io/#/experience/carvalueestimator](https://bchharaw.github.io/#/experience/carvalueestimator)

## Installation
To run this project, make sure you have the following packages installed:
- TensorFlow
- NumPy
- Pandas

You can install these packages by running the following command:
`pip install tensorflow numpy pandas`


## Usage
To use the project, follow these steps:
1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Update the paths to the `test.csv` and `train.csv` files in the code to match your system.
4. Run the `car_value_estimator.py` file.
5. The predicted car values will be displayed in the console.

## Data
The dataset used for this project is the CarDekho Vehicle Dataset. It contains information about various cars, including features such as fuel type, seller type, transmission type, owner, year, kilometer driven, mileage, engine size, power, and number of seats. The dataset was cleaned and preprocessed in Google Sheets to remove any inconsistencies or missing values.

## Model
The linear regression model in this project is implemented using TensorFlow's Estimator API. The model takes a combination of categorical and numerical features as inputs. Categorical features are encoded using vocabulary lists, while numerical features are treated as floating-point values. The model is trained using the training data and evaluated using the testing data.
