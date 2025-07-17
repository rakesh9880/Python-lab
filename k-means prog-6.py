import pandas as pd
data =pd.read_csv("C:/Users/HP/Downloads/Mall_Customers.csv")
print(data.head())
# print(data.tail())

x = data[['Annual Income (k$)','Spending Score (1-100)']]

from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()

x_scaled=scalar.fit_transform(x)
print("Features scaled")

from sklearn.cluster import KMeans
inertia =[]
for k in range(1,11):
    model=KMeans(n_clusters=k,random_state=42)
    model.fit(x_scaled)
    inertia.append(model.inertia_)

kmeans=KMeans(n_clusters=2,random_state=42)
data['clusters']=kmeans.fit_predict(x_scaled)

centroids=scalar.inverse_transform(kmeans.cluster_centers_)
print("\n Centroids")
for i,c in enumerate(centroids):
    print(f"Cluster{i}:Income={c[0]},score={c[1]}")
print("\n Cluster counts")
print(data['clusters'].value_counts().sort_index())

import matplotlib.pyplot as plt
plt.scatter(x_scaled[:,0],x_scaled[:,1],c=data['clusters'])
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c='black',marker='X')
plt.xlabel("Income")
plt.ylabel("spending Score")
plt.title("K-Means Clustering")
plt.show()
