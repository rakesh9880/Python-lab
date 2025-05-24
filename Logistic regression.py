import pandas as pd


data = pd.read_csv("D:/iris.csv")
# print(df.head())
# print(df.describe())
# print(df.info)
# print(df.isnull().sum())

x=data.drop("target",axis=1)
y=data["target"]


from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
x_scalar = scalar.fit_transform(x)
print("feature processed")

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scalar, y,train_size=0.8, test_size=0.2,random_state=50)
print("model is splited " )

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=200)
model.fit(x_train,y_train)
print("logistic regression model trained sucessfully")

train_score=model.score(x_train,y_train)
print("model accuracy is",train_score)

import numpy as np
new_sample=np.array([[6,3,4.8,1.8]])
new_scaled=scalar.transform(new_sample)
prediction=model.predict(new_scaled)
print(prediction[0])
