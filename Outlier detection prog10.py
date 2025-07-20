#10.Write a python program to implement Outlier detection using LOF(Local Outlier Factor).

import pandas as pd
data=pd.read_csv("C:/Users/HP/Downloads/forestfires.csv")

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['month']=le.fit_transform(data['month'])
data['day']=le.fit_transform(data['day'])

x=data.drop(columns=['area'])

from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
x_scaled=scalar.fit_transform(x)

from sklearn.neighbors import LocalOutlierFactor
lof=LocalOutlierFactor(n_neighbors=20,contamination=0.05)
outlier_labels=lof.fit_predict(x_scaled)

data['outlier']=outlier_labels

print("Number of outliers detected:",sum(outlier_labels==-1))
print(data[data['outlier']==-1].head())

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.countplot(x='outlier',data=data)
plt.title("LOF Outlier Detection on Forest Fires Dataset")
plt.xlabel("Outlier(-1) vs Inlier(1)")
plt.show()
