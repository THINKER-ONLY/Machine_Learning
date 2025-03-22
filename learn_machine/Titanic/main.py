import pandas as pd

from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('data/train.csv')

features = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
x = data[features].copy() 
y = data["Survived"]

# 处理缺失值
x["Age"].fillna(x["Age"].median(), inplace=True)
x["Fare"].fillna(x["Fare"].median(), inplace=True)
x["Embarked"].fillna("S", inplace=True)  # 填充最常见的登船港口
x = pd.get_dummies(x)

model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
model.fit(x, y)

test_data = pd.read_csv('data/test.csv')
test_x = test_data[features].copy() 

missing_cols = set(x.columns) - set(test_x.columns)
for col in missing_cols:
    test_x[col] = 0
test_x = test_x[x.columns]

predictions = model.predict(test_x)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('data/my_submission.csv', index=False)