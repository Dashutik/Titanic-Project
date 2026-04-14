import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score




train = pd.read_csv('train.csv')
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
features_test = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']

X = pd.get_dummies(train[features])
y = train['Survived'] #результат к которому мы стремимся

X['Age'] = X['Age'].fillna(X['Age'].median())

model = RandomForestClassifier(n_estimators=100) #создаем лес из 100 деревьев


# Проверяем на тестовой части (которую модель не видела)
test = pd.read_csv('test.csv')

X_test = pd.get_dummies(test[features]) #превращаем текстовые данные в числа 1-0
X_test['Age'] = X_test['Age'].fillna(X_test['Age'].median())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)


predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")

plt.figure(figsize=(10, 6))

importance = model.feature_importances_
feature_names = X.columns

plt.barh(feature_names, importance, color='steelblue')
plt.xlabel('Важность')
plt.title('Что важнее всего для модели')

plt.tight_layout()
plt.savefig('my_graph.png', dpi=100)


