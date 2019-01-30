#IMPORTS
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import csv
import os
import matplotlib.pyplot as plt


X = np.empty(4)
y = np.zeros(32)
abbreviations = {}
numtodivide = {}

#load abbreviations to make data parsing easier
with open('abbrev.csv', mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        fullname = ""
        if "Los" in rows[1]:
            fullname = "LA " + rows[1].split(" ")[-1]
        elif "New York" in rows[1]:
            fullname = "NY " + rows[1].split(" ")[-1]
        else:
            fullname = " ".join(rows[1].split(" ")[:-1])
        abbreviations[fullname] = rows[0]

#print(abbreviations)

#load standings for each team - 2018 only currently
for file in os.listdir("wins"):
    if file.endswith("wins2018.csv"):
        f = np.genfromtxt("wins/" + file, delimiter=',', dtype=None, encoding=None)
        for row in f:
            try:
                row[0] = abbreviations[row[0]]
            except Exception as e:
                continue
            X = np.vstack((X, np.append(row[0], row[1].split('-'))))
X = np.delete(X, (0), axis=0)

#print(X)

#load QB time to throw - 2018 only currently
for file in os.listdir("nextgen"):
    if file.endswith("qb2018.csv"):
        f = np.genfromtxt("nextgen/" + file, delimiter=',', dtype=None, encoding=None)
        for row in f:
            index = np.where(X == row[1])
            if (y[index[0]][0] == 0):
                np.put(y, np.array(index[0]), row[2])
                numtodivide[row[1]] = 1
            else:
                y[index[0]] = y[index[0]] + row[2]
                numtodivide[row[1]] = numtodivide[row[1]] + 1

#fit regression curve
regr = linear_model.LinearRegression()
train_x = X[:,1].reshape(-1,1)[:25]
train_y = y.astype(float)[:25]
test_x = X[:,1].reshape(-1,1).astype(float)[:-25]
test_y = y.astype(float)[:-25]
regr.fit(train_x,train_y)

#generate predictions and measure success
pred_y = regr.predict(test_x)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(test_y, pred_y))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(test_y, pred_y))

# Plot outputs
plt.scatter(test_x, test_y,  color='black')
plt.plot(test_x, pred_y, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
