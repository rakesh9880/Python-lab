import pandas as pd
data = pd.read_csv("D:/iris_naivebayes.csv")

x = data.drop("target",axis=1)
y = data["target"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=48)
print("model splited successfully")

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
print("model successfully" )

train_accuracy = knn.score(x_train,y_train)
test_accuracy = knn.score(x_test,y_test)
print("model train accuracy is : ",train_accuracy)
print("model test accuracy is : ",test_accuracy)

y_pred = knn.predict(x_test)
correct_predication = (y_pred == y_test)
wrong_predication = (y_pred != y_test)
print("Wrong Predication ")
print(x_test[wrong_predication])

print("Correct Predication ")
print(x_test[correct_predication])
