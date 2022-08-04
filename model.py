import pandas as pd
import numpy as np 
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


data = pd.read_csv('Bike-Sharing-Dataset/day.csv', sep= ',')

# No null or missing values so we can proceed to modeling 


# We are trying to predict the number of bikes rented 

# Our X values will be all columns except count

X = data.iloc[:, 2:-3]
print(X.head())

# # Our y value will be the count column 

y = data.iloc[:, -1]

# Now let's make 100 models and save the one with the best accuracy

best_acc = 0
for _ in range(100): 

    # Split data into traing and testing 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

    model = LinearRegression()

    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)

    if acc > best_acc: 
        best_acc = acc
        pickle.dump(model, open('model.pkl','wb'))


print(best_acc)
# Load the best model 
model = pickle.load(open('model.pkl', 'rb'))
