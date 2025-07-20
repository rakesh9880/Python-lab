#8.Write a python program to implement support vector machine algorithm.
import pandas as pd
data=pd.read_csv("C:/Users/HP/Downloads/give_me_credit.csv")
data=data.dropna()

x=data.drop("SeriousDlqin2yrs",axis=1)
y=data["SeriousDlqin2yrs"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,train_size=0.8,random_state=42)

from sklearn.svm import SVC
model=SVC(kernel='rbf',C=1.0)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

from sklearn.metrics import classification_report,accuracy_score

print("Accuracy:",accuracy_score(y_test,y_pred))
print("\n classification report\n",classification_report(y_test,y_pred))

correct=x_test[y_test==y_pred]
wrong=x_test[y_test!=y_pred]

print("\n Top 5 correct prediction:\n",correct.head())
print("\n Top 5 Wrong prediction:\n",wrong.head())
