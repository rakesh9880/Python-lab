import pandas as pd


data=pd.read_csv("C:/Users/HP/Desktop/BostonHousing_pgm1.csv")

#print(data.head(10))
#print(data.isnull())
#print(data.isnull().sum())

X=data[['rm']]
y=data['medv']

#split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

from sklearn.linear_model import LinearRegression



#training
model=LinearRegression()
model.fit(X_train, y_train)


#predict
y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error,r2_score

#evaluate
print("R2:",r2_score(y_test,y_pred))
print("MSE",mean_squared_error(y_test,y_pred))


#plot
import matplotlib.pyplot as plt
plt.scatter(X_test,y_test,color='blue',label='Actual')
plt.plot(X_test,y_pred,color='red',linewidth=2,label='predictd')
plt.xlabel("Rooms per house")
plt.ylabel("House price")
plt.title("Linear Regression")
plt.legend()
plt.show()
