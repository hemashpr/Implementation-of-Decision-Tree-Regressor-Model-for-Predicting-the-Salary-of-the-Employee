# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1. Import the required libraries and load the Salary dataset using Pandas.
2. Preprocess the dataset by handling missing values and converting categorical features into numerical form.
3. Separate the input features and the employee salary as the target variable.
4. Split the dataset into training and testing sets and train the Decision Tree Regressor model.
5. Predict employee salaries and evaluate the model using appropriate regression performance metrics.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Hemash K
RegisterNumber:  21225050015
*/

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("Salary.csv")

# Display dataset
print(data.head())

# Select input feature and target variable
X = data[["Level"]]
y = data["Salary"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\nMean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Display results
results = pd.DataFrame({
    "Actual Salary": y_test.values,
    "Predicted Salary": y_pred
})

print("\nActual vs Predicted Salary:")
print(results)
```

## Output:
<img width="890" height="247" alt="image" src="https://github.com/user-attachments/assets/164dbf09-3a85-46f8-9df0-1acda1607f08" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
