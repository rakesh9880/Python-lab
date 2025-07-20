#9.Write a python program to demonstrate Random Forest Algorithm using an appropriate dataset.
import pandas as pd
data=pd.read_csv("C:/Users/HP/Downloads/iris_naivebayes.csv")

x=data.drop(columns = ["target"])
y=data["target"]

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y_encoded=le.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y_encoded,test_size=0.2,train_size=0.8,random_state=58)

from sklearn.ensemble import RandomForestClassifier
rf_model=RandomForestClassifier(n_estimators=10,random_state=42)
rf_model.fit(x_train,y_train)

sample_index=24
sample=x_test.iloc[sample_index].values.reshape(1,-1)

tree_preds=[tree.predict(sample)[0] for tree in rf_model.estimators_]


from collections import Counter
vote_counts=Counter(tree_preds)

label_votes={le.inverse_transform([int(k)])[0]:v for k,v in vote_counts.items()}

print("\n Class Votes:")
for label,count in label_votes.items():
    print(f"{label}:{count} vote(s)")

majority_encoded,_=vote_counts.most_common(1)[0]
majority_label=le.inverse_transform([int(majority_encoded)])[0]

true_label=le.inverse_transform(([int(y_test[sample_index])]))[0]

print(f"\n Final prediction(Majority Vote):{majority_label}")
print(f"Actual Label:{true_label}")
