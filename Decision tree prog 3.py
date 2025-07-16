import pandas as pd

data = pd.read_csv("D:/PlayTennis.csv")
# print(df.head())
# print(df.describe())
# print(df.info)
# print(df.isnull().sum())

for col in data.columns[:-1]:
    data[col] = data[col].astype('category')
    mapping=dict(enumerate(data[col].cat.categories))
    print(f"{col}:{mapping}")
    data[col]=data[col].cat.codes
    print("converted")


from sklearn.model_selection import train_test_split
x=data.drop("Play Tennis",axis=1)
y=data["Play Tennis"]
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size= 0.8,test_size= 0.2,random_state= 58)
print("spilted data")

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(criterion="entropy",random_state=58)
model.fit(x_train,y_train)
print("trained data")
print("acc train\n",model.score(x_train,y_train))
print("acc test\n",model.score(x_test,y_test))

sample=pd.DataFrame(data=[[1,0,1,1]],columns=x.columns)
p=model.predict(sample)
print("prediction",p)

