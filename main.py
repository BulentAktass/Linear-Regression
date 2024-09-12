import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings("ignore")
salary = pd.read_csv("Salary_Data.csv")

print(salary.head())

sns.scatterplot(x='YearsExperience', y='Salary', data=salary)
plt.title('Scatter plot between Years of Experience and Salary')
plt.show()

sns.heatmap(salary.corr(), cmap="YlGnBu", annot=True)
plt.show()

sns.histplot(salary['Salary'], kde=True)
plt.title('Histogram of Salary')
plt.show()

print("*******")
print(salary.describe())
print("*******")

print(salary.info())
print("*******")

X = salary["YearsExperience"]
y = salary["Salary"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train, X_train_sm).fit()

print("***************************")
print(lr.params)
print("***************************")
print(lr.summary())

X_test_sm = sm.add_constant(X_test)
y_pred = lr.predict(X_test_sm)

print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R-Squared: ", r2_score(y_test, y_pred))

plt.scatter(X_train, y_train)
plt.plot(X_train, 25202.887786 + 9731.2038 * X_train, 'r')
plt.title("Actual salary vs Predicted salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

df_check = pd.DataFrame({'Actual': y_test[:10].values, 'Predicted': y_pred[:10].ravel()})
print(df_check)

y_train_pred = lr.predict(X_train_sm)
res = (y_train - y_train_pred)

fig = plt.figure()
sns.distplot(res, bins=15)
fig.suptitle('Error Terms', fontsize=15)
plt.xlabel('y_train - y_train_pred', fontsize=15)
plt.show()
